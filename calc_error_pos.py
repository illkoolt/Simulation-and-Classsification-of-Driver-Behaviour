import pandas as pd
import numpy as np
from collections import defaultdict

# === IDM Acceleration Function ===
def calculate_idm_acc(v, v0_mps, T, s, s0, a, b, delta_v_mps):
    """Calculate IDM acceleration with numerical stability checks"""
    # Parameter validation
    if (not np.isfinite(v) or not np.isfinite(v0_mps) or v0_mps <= 0.1 or s <= 0.1 or a <= 0.01 or b <= 0.01):
        return 0.0
    
    try:
        # Calculate sqrt(a*b) with bounds
        sqrt_ab = np.sqrt(a*b)
        if sqrt_ab < 0.1:
            return 0.0
        
        # Calculate s_star with bounds checking
        term1 = s0
        term2 = v * T
        term3 = (v * delta_v_mps) / (2 * sqrt_ab)
        
        # Check each term for overflow
        for term in [term1, term2, term3]:
            if not np.isfinite(term) or abs(term) > 1e6:
                return 0.0
                
        s_star = term1 + term2 + term3
        if s_star <= 0 or not np.isfinite(s_star):
            return 0.0
        
        # Calculate velocity term with bounds
        vel_ratio = v / v0_mps
        if abs(vel_ratio) > 10:  # Unrealistic velocity ratio
            return 0.0
            
        vel_term = vel_ratio**4
        if not np.isfinite(vel_term):
            vel_term = 0.0
        
        # Calculate spacing term with bounds
        spacing_ratio = s_star / s
        if spacing_ratio < 0 or not np.isfinite(spacing_ratio):
            return 0.0
            
        spacing_term = spacing_ratio**2
        if not np.isfinite(spacing_term):
            spacing_term = 0.0
        
        # Final acceleration calculation with bounds
        idm_acc = a * (1 - vel_term - spacing_term)
        
        # Clip to reasonable physical limits
        return np.clip(idm_acc, -10.0, 10.0)
        
    except (ValueError, ZeroDivisionError, RuntimeWarning):
        return 0.0

def load_parameters(csv_path):
    """Load parameters from CSV file"""
    df = pd.read_csv(csv_path)
    all_data = []
    
    for _, row in df.iterrows():
        rec_id = row['rec_id']
        driver_id = row['driver_id']
        s0 = 2.0  # Default value, can be changed if available
        
        # Prepare parameters for each algorithm
        params = {
            'ga': {
                'T': row['GA_T'],
                'a': row['GA_a'],
                'b': row['GA_b'],
                'v0': row['GA_v0']
            },
            'ml': {
                'T': row['MLE_T'],
                'a': row['MLE_a'],
                'b': row['MLE_b'],
                'v0': row['MLE_v0']
            },
            'de': {
                'T': row['DE_T'],
                'a': row['DE_a'],
                'b': row['DE_b'],
                'v0': row['DE_v0']
            }
        }
        
        all_data.append({
            'recId': rec_id,
            'driverId': driver_id,
            's0': s0,
            'params': params
        })
    
    return all_data

def load_frame_data(csv_path):
    """Load frame data from CSV file"""
    df = pd.read_csv(csv_path)
    frame_data = defaultdict(list)
    
    for _, row in df.iterrows():
        rec_id = row['recId']
        driver_id = row['driverId']
        key = (rec_id, driver_id)
        
        frame_data[key].append({
            'frame': row['frame'],
            'v': row['v'],
            'delta_v': row['delta_v'],
            's': row['s'],
            'XAcceleration': row['XAcceleration']
        })
    
    return frame_data

def calculate_position_errors(parsed_data, frame_data):
    """Calculate position errors for all drivers and algorithms"""
    dt = 1 / 25  # 25 FPS
    driver_results = []
    
    for driver_data in parsed_data:
        rec_id = driver_data['recId']
        driver_id = driver_data['driverId']
        s0 = driver_data['s0']
        params = driver_data['params']
        
        # Get frame data for this driver
        key = (rec_id, driver_id)
        frames = frame_data.get(key, [])
        if len(frames) < 2:  # Need at least 2 frames to calculate movement
            continue
            
        # Sort frames by frame number
        frames = sorted(frames, key=lambda x: x['frame'])
        
        # Calculate for each algorithm
        for algo, algo_params in params.items():
            # Initialize simulation
            sim_pos = [0.0]
            sim_vel = frames[0]['v']
            actual_pos = [0.0]
            actual_vel = frames[0]['v']
            
            valid_simulation = True
            
            # Run simulation for each frame (skip first frame as starting point)
            for i in range(1, len(frames)):
                frame = frames[i]
                
                # Calculate IDM acceleration with safety checks
                try:
                    idm_acc = calculate_idm_acc(
                        sim_vel, algo_params['v0'], algo_params['T'],
                        frame['s'], s0, algo_params['a'], algo_params['b'], frame['delta_v']
                    )
                    
                    # Update simulated position
                    sim_vel += idm_acc * dt
                    sim_pos.append(sim_pos[-1] + sim_vel * dt)
                    
                    # Update actual position
                    actual_vel += frame['XAcceleration'] * dt
                    actual_pos.append(actual_pos[-1] + actual_vel * dt)
                    
                except (ValueError, ZeroDivisionError):
                    valid_simulation = False
                    break
            
            if not valid_simulation or len(sim_pos) < 2:
                continue
                
            # Calculate errors with safety checks
            try:
                sim_pos = np.array(sim_pos)
                actual_pos = np.array(actual_pos)
                squared_errors = (sim_pos - actual_pos)**2
                
                # Filter out any NaN or inf values
                valid_errors = squared_errors[np.isfinite(squared_errors)]
                if len(valid_errors) == 0:
                    continue
                
                rmse = np.sqrt(np.mean(valid_errors))
                avg_pos = np.mean(np.abs(actual_pos[np.isfinite(actual_pos)]))
                
                # Avoid division by zero
                if avg_pos < 1e-5:
                    error_percent = 0.0
                else:
                    error_percent = 100 * rmse / avg_pos
                
                # Ensure error_percent is finite
                if not np.isfinite(error_percent):
                    continue
                    
                driver_results.append({
                    'recId': rec_id,
                    'driverId': driver_id,
                    'algorithm': algo.upper(),
                    'rmse': rmse,
                    'error_percent': error_percent,
                    'T': algo_params['T'],
                    'a': algo_params['a'],
                    'b': algo_params['b'],
                    'v0': algo_params['v0']
                })
                
            except (ValueError, ZeroDivisionError):
                continue
    
    return driver_results

def print_and_save_results(driver_results):
    """Print and save the results"""
    algo_names = {
        'GA': 'Genetic Algorithm',
        'ML': 'Maximum Likelihood',
        'DE': 'Differential Evolution'
    }
    
    # Print results by algorithm
    for algo in ['GA', 'ML', 'DE']:
        algo_results = [r for r in driver_results if r['algorithm'] == algo]
        if not algo_results:
            print(f"\n=== {algo_names[algo]} ===")
            print("No valid results for this algorithm")
            continue
            
        print(f"\n=== {algo_names[algo]} Position Errors ===")
        
        #Print individual driver errors
        # for result in sorted(algo_results, key=lambda x: (x['recId'], x['driverId'])):
        #     print(f"rec {result['recId']}, id {result['driverId']} = {result['error_percent']:.2f}%")
        
        # Calculate and print summary statistics
        error_percents = [r['error_percent'] for r in algo_results if np.isfinite(r['error_percent'])]
        
        print("\n=== Summary Statistics ===")
        print(f"Total drivers analyzed: {len(error_percents)}")
        if error_percents:
            print(f"Mean error percentage: {np.mean(error_percents):.2f}%")
            print(f"Median error percentage: {np.median(error_percents):.2f}%")
            print(f"Minimum error: {np.min(error_percents):.2f}%")
            print(f"Maximum error: {np.max(error_percents):.2f}%")
            print(f"Standard deviation: {np.std(error_percents):.2f}%")
        else:
            print("No valid error percentages available")
    
    # # Save to CSV
    # if driver_results:
    #     results_df = pd.DataFrame(driver_results)
    #     results_df.to_csv('position_errors_results.csv', index=False)
    #     print("\nResults saved to 'position_errors_results.csv'")
    # else:
    #     print("\nNo valid results to save")

def main():
    # Load parameters from CSV
    parsed_data = load_parameters('idm_all_parameters.csv')
    
    # Load frame data from CSV
    frame_data = load_frame_data('idm_output_driver_frames.csv')
    
    # Calculate position errors
    driver_results = calculate_position_errors(parsed_data, frame_data)
    
    # Print and save results
    print_and_save_results(driver_results)

if __name__ == "__main__":
    main()