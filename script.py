import pandas as pd
import numpy as np
import random
import concurrent.futures
from scipy.optimize import minimize, differential_evolution
import csv
from collections import Counter

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import time 

REC_IDS = list(range(1, 61)) #[1-60] 

#=========================DATA=========================#
# load the dataset from folder
META_DATA_PATH = 'tracks_meta'
RAW_DATA_PATH = '241202_highD_lcTrainingTestForOptimisationData_di.pkl'
MAX_REC_ID = 1

#=======================================================#
def load_and_prepare_data(rec_ids):
    # load main pickle file
    raw_data = pd.read_pickle(RAW_DATA_PATH)
    
    # load and combine all relevant meta files
    meta_dfs = []
    for rec_id in rec_ids:
        meta_file = os.path.join(META_DATA_PATH, f'{rec_id:02d}_tracksMeta.csv')
        if os.path.exists(meta_file):
            meta_df = pd.read_csv(meta_file)
            meta_df['recId'] = rec_id  # added recording ID
            meta_dfs.append(meta_df)
    
    vec_type_data = pd.concat(meta_dfs, ignore_index=True)
    
    # filter raw data by recording IDs
    data_list = [
        df for df in raw_data.values()
        if df['recId'].iloc[0] in rec_ids
    ]
    data = pd.concat(data_list, ignore_index=True)
    
    # rename columns
    data = data.rename(columns={
        'frame': 'frame',
        'egoVehId': 'id',
        'speedEgoVeh': 'xVelocity',
        'accelEgoVeh': 'xAcceleration',
        'distEgoVehToLeadVehOnCurrentLane': 'dhw',
        'leadVehOnCurrentLane': 'precedingId',
    })
    
    #  preceding vehicle velocity
    data['precedingXVelocity'] = data['xVelocity'] - data['speedDifferenceEgoVehToLeadVehOnCurrentLane'] 
   
    #merge with vehicle type data
    vec_type_data_subset = vec_type_data[['recId', 'id', 'class']]
    data = pd.merge(data, vec_type_data_subset, how='left', on=['recId', 'id'])
    
    data['class'] = data['class'].fillna('Car') # fill missing vehicle class with 'Car' 


    # to run only for driver ID 77
    # data = data[(data['id'] >= 70) & (data['id'] <= 200)]
    # data = data[(data['recId'] == 5) & (data['id'] == 813)]

    # Get first 10 unique driver IDs
    first_occurrences = data.drop_duplicates(subset=['id'], keep='first')
    first_10_drivers = first_occurrences['id'].head(1).tolist()

    # Now filter the complete dataset for these 10 drivers
    data = data[data['id'].isin(first_10_drivers)]

    return data
#=======================================================#

#===========CALL EACH DRIVER===========#
def calc_idm_params(data):
    # grouped by both recId and id to ensure uniqueness
    grouped = data.groupby(['recId', 'id'])
    
    print(f"Total unique drivers to process: {len(grouped)}")
    recid_counts = Counter(rec_id for (rec_id, _), _ in grouped)
    for rec_id, count in recid_counts.items():
        print(f"recId {rec_id}: {count} drivers")
    
    idm_params = {}
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_driver, driver_key, driver_data): driver_key
            for driver_key, driver_data in grouped
        }
        
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            rec_id, driver_id = futures[future]
            try:
                idm_params[(rec_id, driver_id)] = future.result()
                print(f"Completed {i}/{len(grouped)} (Rec {rec_id} Driver {driver_id})")
            except Exception as exc:
                print(f'Rec {rec_id} Driver {driver_id} failed: {exc}')
    
    return idm_params

#===========================================#
#Manually fill PID with 0
#===========================================#
def manual_ffill(series, value_to_replace=0):
    prev_val = None
    new_series = series.copy()
    for i in range(len(new_series)):
        if new_series.iloc[i] == value_to_replace:
            if prev_val is not None:
                new_series.iloc[i] = prev_val
        else:
            prev_val = new_series.iloc[i]
    return new_series


#===========================================#
#Process driver ID separate
#===========================================#
def process_driver(driver_id, driver_data):
    # Direction check 
    if driver_data['xVelocity'].mean() < 0:
        driver_data['xVelocity'] = -driver_data['xVelocity']
        driver_data['precedingXVelocity'] = -driver_data['precedingXVelocity']
        driver_data['xAcceleration'] = -driver_data['xAcceleration']

    s0 = 2  # Minimum spacing

    # check if missing data
    # driver_data['precedingId'] = manual_ffill(driver_data['precedingId'])
    # driver_data['precedingXVelocity'] = manual_ffill(driver_data['precedingXVelocity'])
    # driver_data['dhw'] = manual_ffill(driver_data['dhw'])
    driver_data.loc[driver_data['precedingId'] == 0, 'dhw'] = 250.0

    # calc IDM inputs for the driver
    idm_results = []
    for _, row in driver_data.iterrows():
        idm_results.append({
            'frame': row['frame'],
            'v': row['xVelocity'],
            's': max(row['dhw'], 0.0001),
            'delta_v_mps': 1.0 if row['dhw'] == 250.0 else row['xVelocity'] - row['precedingXVelocity'],
            'precedingId': row['precedingId'],
            'precedingXVelocity': row['precedingXVelocity'],
            'XAcceleration': row['xAcceleration']
        })
 

    # optimization for this driver (single segment)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Start timing for GA
        start_ga = time.time()
        ga_future = executor.submit(genetic_algorithm, idm_results, s0, driver_data.iloc[0]['class'])
        ga_params = ga_future.result()
        end_ga = time.time()
        ga_time = end_ga - start_ga
        
        # Start timing for MLE
        start_mle = time.time()
        mle_future = executor.submit(maximum_likelihood_estimation, idm_results, s0, driver_data.iloc[0]['class'])
        mle_params = mle_future.result()
        end_mle = time.time()
        mle_time = end_mle - start_mle
        
        # Start timing for DE
        start_de = time.time()
        de_future = executor.submit(differential_evolution_optimizer, idm_results, s0, driver_data.iloc[0]['class'])
        de_params = de_future.result()
        end_de = time.time()
        de_time = end_de - start_de

    print(f"\nAlgorithm Runtimes for driver {driver_id}:")
    print(f"Genetic Algorithm: {ga_time:.4f} seconds")
    print(f"Maximum Likelihood Estimation: {mle_time:.4f} seconds")
    print(f"Differential Evolution: {de_time:.4f} seconds")

    for result in idm_results:
        # GA calc
        idm_ga, s_star_ga = calculate_idm_acc(result['v'], ga_params[3], ga_params[0], 
                                            result['s'], s0, ga_params[1], ga_params[2], 
                                            result['delta_v_mps'])
        # MLE calc
        idm_mle, s_star_mle = calculate_idm_acc(result['v'], mle_params[3], mle_params[0], 
                                              result['s'], s0, mle_params[1], mle_params[2], 
                                              result['delta_v_mps'])
        # DE calc
        idm_de, s_star_de = calculate_idm_acc(result['v'], de_params[3], de_params[0], 
                                            result['s'], s0, de_params[1], de_params[2], 
                                            result['delta_v_mps'])
        
        result.update({
            'idm_ga': idm_ga,
            'idm_mle': idm_mle,
            'idm_de': idm_de,
            's_star_ga': s_star_ga,
            's_star_mle': s_star_mle,
            's_star_de': s_star_de
        })
        
    return {
        's0': s0,
        'idm_results': idm_results,
        'GA_params': {
            'T': ga_params[0], 'a': ga_params[1], 'b': ga_params[2], 'v0_mps': ga_params[3]
        },
        'MLE_params': {
            'T': mle_params[0], 'a': mle_params[1], 'b': mle_params[2], 'v0_mps': mle_params[3]
        },
        'DE_params': {
            'T': de_params[0], 'a': de_params[1], 'b': de_params[2], 'v0_mps': de_params[3]
        },
        'class': driver_data.iloc[0]['class']
    }

#=================================================================#
def calculate_idm_acc(v, v0_mps, T, s, s0, a, b, delta_v_mps):
    if a <= 0 or b <= 0 or v0_mps <= 0 or s <= 1e-3:
        return 0.0, float('inf')

    sqrt_ab = np.sqrt(a * b)
    if not np.isfinite(sqrt_ab) or sqrt_ab < 1e-3:
        return 0.0, float('inf')

    term = (v * delta_v_mps) / (2 * sqrt_ab)
    if not np.isfinite(term):
        return 0.0, float('inf')

    s_star = s0 + v * T + term
    if not np.isfinite(s_star) or s_star < 1e-3:
        s_star = 1e-3

    v_ratio = v / max(v0_mps, 1e-3)
    s_ratio = s_star / max(s, 1e-3)

    if v_ratio > 1e2:
        v_ratio = 1e2
    if s_ratio > 1e2:
        s_ratio = 1e2

    idm_acc = a * (1 - v_ratio**4 - s_ratio**2)

    if not np.isfinite(idm_acc):
        return 0.0, s_star

    return idm_acc, s_star
#=================================================================#

#=============================================================================#
def genetic_algorithm(frames, s0, vehicle_class, pop_size=100, generations=150, mutation_rate=0.15):
    T_range = [0.5, 5.0]
    a_range = [0.1, 5.0]
    b_range = [0.1, 5.0]
    
    v_max = max(result['v'] for result in frames)

    global_bounds = {
        'Car': (80 / 3.6, 150 / 3.6),
        'Truck': (60 / 3.6, 120 / 3.6)
    }

    min_global, max_global = global_bounds.get(vehicle_class, (60 / 3.6, 150 / 3.6))
    v0_lower = max(min_global, v_max)
    v0_upper = max(max_global, v_max * 1.05)  # 5% buffer above observed v_max
    v0_range = [v0_lower, v0_upper]

    assert v0_lower < v0_upper, f"Invalid v0 bounds: [{v0_lower}, {v0_upper}] for v_max={v_max}, class={vehicle_class}"
    
    # init population with some good guesses
    population = []
    for _ in range(pop_size):
        population.append([
            random.uniform(*T_range),
            random.uniform(*a_range),
            random.uniform(*b_range),
            random.uniform(*v0_range)
        ])
    
    # track best individual across all generations
    global_best = None
    global_best_fitness = float('inf')

    for generation in range(generations):
        
        fitness_scores = []
        # evaluate fitness
        for ind in population:
            current_fitness = fitness(ind, frames, s0, vehicle_class)
            fitness_scores.append(current_fitness)
            if current_fitness < global_best_fitness:
                global_best = ind
                global_best_fitness = current_fitness

        # tournament selection
        selected = []
        tournament_size = 3
        while len(selected) < pop_size//2:
            candidates = random.sample(list(zip(fitness_scores, population)), tournament_size)
            winner = min(candidates, key=lambda x: x[0])[1]
            selected.append(winner)

        # crosscover and mutation
        new_population = selected.copy()
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(selected, 2)
            child = [
                random.choice([parent1[i], parent2[i]]) if random.random() > 0.5 
                else (parent1[i] + parent2[i])/2 
                for i in range(4)
            ]
            
            # adaptive mutation
            for i in range(4):
                if random.random() < mutation_rate:
                    ranges = [T_range, a_range, b_range, v0_range][i]
                    child[i] = np.clip(child[i] + random.gauss(0, 0.1*(ranges[1]-ranges[0])),
                                      ranges[0], ranges[1])

            new_population.append(child)

        population = new_population

    return global_best  # return the best ever found, not just last generation
#=============================================================================#

#=============================================================================#
def maximum_likelihood_estimation(frames, s0, vehicle_class):
    v_max = max(result['v'] for result in frames)

    global_bounds = {
        'Car': (80 / 3.6, 150 / 3.6),
        'Truck': (60 / 3.6, 120 / 3.6)
    }

    min_global, max_global = global_bounds.get(vehicle_class, (60 / 3.6, 150 / 3.6))

    v0_lower = max(min_global, v_max)
    v0_upper = max(max_global, v_max * 1.05)  # 5% buffer above observed v_max
    v0_range = [v0_lower, v0_upper]
    
    bounds = [
        (0.5, 5.0),    # T
        (0.1, 5.0),    # a
        (0.1, 5.0),    # b 
        (v0_range)
    ]

    assert v0_lower < v0_upper, f"Invalid v0 bounds: [{v0_lower}, {v0_upper}] for v_max={v_max}, class={vehicle_class}"
    
    # multiple starting points to avoid local minimum
    best_params = None
    best_error = float('inf')
    
    for _ in range(3):  # try 3 different starting points
        x0 = [
            random.uniform(low, high)
            for (low, high) in bounds
        ]
        
        result = minimize(
            fun=fitness,
            x0=x0,
            args=(frames, s0, vehicle_class),
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 100}
        )
        
        if result.fun < best_error:
            best_error = result.fun
            best_params = result.x
    
    return best_params

#=============================================================================#
def differential_evolution_optimizer(frames, s0, vehicle_class):
    v_max = max(result['v'] for result in frames)

    global_bounds = {
        'Car': (80 / 3.6, 150 / 3.6),
        'Truck': (60 / 3.6, 120 / 3.6)
    }

    min_global, max_global = global_bounds.get(vehicle_class, (60 / 3.6, 150 / 3.6))

    v0_lower = max(min_global, v_max)
    v0_upper = max(max_global, v_max * 1.05)  # 5% buffer above observed v_max
    v0_range = [v0_lower, v0_upper]
    
    bounds = [
        (0.5, 5.0),    # T
        (0.1, 5.0),     # a
        (0.1, 5.0),     # b
        (v0_range)    # range of v0
    ]

    assert v0_lower < v0_upper, f"Invalid v0 bounds: [{v0_lower}, {v0_upper}] for v_max={v_max}, class={vehicle_class}"
    
    result = differential_evolution(
        func=fitness,
        bounds=bounds,
        args=(frames, s0, vehicle_class),
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42
    )
    return result.x


def fitness(params, frames, s0, vehicle_class):
    T, a, b, v0 = params

    if any(p < 0 for p in [a, b]) or v0 < 1:
        return float('inf')

    dt = 1 / 25  # 25 FPS
    abs_error_sq = []
    rel_error_sq = []
    mixed_error_terms = []
    s_data_total = 0.0

    try:
        follower_v = frames[0]['v']
        leader_v = frames[0]['precedingXVelocity']
        follower_x = 0.0
        leader_x = frames[0]['s']  # initial observed gap

        for frame in frames:
            s_data = max(frame['s'], 0.01)
            s_data_total += s_data

            delta_v = frame['delta_v_mps']
            acc_idm, _ = calculate_idm_acc(follower_v, v0, T, s_data, s0, a, b, delta_v)

            follower_v += acc_idm * dt
            follower_x += follower_v * dt

            leader_v = frame['precedingXVelocity']
            leader_x += leader_v * dt

            s_sim = max(leader_x - follower_x, 0.01)

            # acc all error terms
            abs_error_sq.append((s_sim - s_data) ** 2)
            rel_error_sq.append(((s_sim - s_data) / s_data) ** 2)
            mixed_error_terms.append(((s_sim - s_data) ** 2) / s_data)

    except:
        return float('inf')

    if not abs_error_sq or s_data_total == 0:
        return float('inf')

    # final formulas metrics
    avg_s_data = s_data_total / len(abs_error_sq)
    
    f_rel = np.sqrt(sum(rel_error_sq) / len(rel_error_sq)) # 1
    f_abs = np.sqrt(sum(abs_error_sq) / len(abs_error_sq)) / avg_s_data
    f_mixed_error = np.sqrt(sum(mixed_error_terms) / len(mixed_error_terms) / avg_s_data)

    return f_rel  # switch of f


#=================================================#
# MAIN with save to file (updated for segments)
#=================================================#
if __name__ == '__main__':
    # load and prepare data for selected recordings
    data = load_and_prepare_data(REC_IDS[:MAX_REC_ID])  
    
    idm_params = calc_idm_params(data)

    with open('idm_output.txt', 'w') as output_file:
        for driver_key in sorted(idm_params.keys()):
            rec_id, driver_id = driver_key
            params = idm_params[driver_key]
            
            #driver header
            output_file.write(f"Recording {rec_id} - Driver ID: {driver_id}, Class: {params['class']}\n")
            output_file.write(f"  s0: {params['s0']:.2f} m\n\n")
            
            # parameters 
            output_file.write("  Genetic Algorithm Parameters:\n")
            output_file.write(f"    T: {params['GA_params']['T']:.2f} s, "
                            f"a: {params['GA_params']['a']:.2f} m/s², "
                            f"b: {params['GA_params']['b']:.2f} m/s², "
                            f"v0: {params['GA_params']['v0_mps']:.2f} m/s\n")
            
            output_file.write("  Maximum Likelihood Parameters:\n")
            output_file.write(f"    T: {params['MLE_params']['T']:.2f} s, "
                            f"a: {params['MLE_params']['a']:.2f} m/s², "
                            f"b: {params['MLE_params']['b']:.2f} m/s², "
                            f"v0: {params['MLE_params']['v0_mps']:.2f} m/s\n")
            
            output_file.write("  Differential Evolution Parameters:\n")
            output_file.write(f"    T: {params['DE_params']['T']:.2f} s, "
                            f"a: {params['DE_params']['a']:.2f} m/s², "
                            f"b: {params['DE_params']['b']:.2f} m/s², "
                            f"v0: {params['DE_params']['v0_mps']:.2f} m/s\n\n")

            # frame data 
            output_file.write("  Frame Data:\n")
            for result in params['idm_results']:
                output_file.write(
                    f"    Frame: {result['frame']:.0f},(PID:{result['precedingId']}=>v:{result['precedingXVelocity']:.2f}); "
                    f"v: {result['v']:.2f}, Δv: {result['delta_v_mps']:.2f}, s: {result['s']:.2f}, s*_GA: {result['s_star_ga']:.2f}, "
                    f"[IDM_GA: {result['idm_ga']:.4f} | IDM_MLE: {result['idm_mle']:.4f} | IDM_DE: {result['idm_de']:.4f} | XAcc: {result['XAcceleration']:.4f} m/s²]\n"
                )
            output_file.write("\n" + "="*80 + "\n\n")


    with open('idm_output_driver_frames.csv', mode='w', newline='') as csv_file:
        fieldnames = [
            'recId', 'driverId', 'frame', 'precedingId',
            'precedingXVelocity', 'v', 'delta_v', 's',
            'idm_ga', 'idm_mle', 'idm_de', 
            's_star_ga', 's_star_mle', 's_star_de',
            'XAcceleration'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for driver_key in sorted(idm_params.keys()):
            rec_id, driver_id = driver_key
            p = idm_params[driver_key]

            for result in p['idm_results']:
                writer.writerow({
                    'recId': rec_id,
                    'driverId': driver_id,
                    'frame': result['frame'],
                    'precedingId': result['precedingId'],
                    'precedingXVelocity': round(result['precedingXVelocity'], 2),
                    'v': round(result['v'], 2),
                    'delta_v': round(result['delta_v_mps'], 2),
                    's': round(result['s'], 2),
                    'idm_ga': round(result['idm_ga'], 4),
                    'idm_mle': round(result['idm_mle'], 4),
                    'idm_de': round(result['idm_de'], 4),
                    's_star_ga': round(result['s_star_ga'], 4),
                    's_star_mle': round(result['s_star_mle'], 4),
                    's_star_de': round(result['s_star_de'], 4),
                    'XAcceleration': round(result['XAcceleration'], 4)
                })

    
    # save all parameters for each algorithm
    with open('idm_all_parameters.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'rec_id', 
            'driver_id', 
            'GA_T', 'GA_a', 'GA_b', 'GA_v0',
            'MLE_T', 'MLE_a', 'MLE_b', 'MLE_v0',
            'DE_T', 'DE_a', 'DE_b', 'DE_v0'
        ])

        for (rec_id, driver_id), params in sorted(idm_params.items()):
            try:
                writer.writerow([
                    rec_id,
                    driver_id,
                    round(params['GA_params']['T'], 3),
                    round(params['GA_params']['a'], 3),
                    round(params['GA_params']['b'], 3),
                    round(params['GA_params']['v0_mps'], 3),
                    round(params['MLE_params']['T'], 3),
                    round(params['MLE_params']['a'], 3),
                    round(params['MLE_params']['b'], 3),
                    round(params['MLE_params']['v0_mps'], 3),
                    round(params['DE_params']['T'], 3),
                    round(params['DE_params']['a'], 3),
                    round(params['DE_params']['b'], 3),
                    round(params['DE_params']['v0_mps'], 3)
                ])
            except KeyError:
                continue