import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

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

# === Load Data === #
frames_df = pd.read_csv('idm_output_driver_frames.csv')
params_df = pd.read_csv('idm_all_parameters.csv')

# Select only GA parameters and rename columns
params_df = params_df[['rec_id', 'driver_id', 'GA_T', 'GA_a', 'GA_b', 'GA_v0']]
params_df.columns = ['recId', 'driverId', 'T', 'a', 'b', 'v0']

# === Settings === #
dt = 1 / 25  # 25 FPS
s0 = 2.0  # minimum spacing, fixed
ERROR_THRESHOLD = 60  # Percentage threshold for printing problematic drivers

# === Function to analyze specific driver ===
def analyze_specific_driver(rec_id, driver_id):
    # Find the specific driver's data
    group = frames_df[(frames_df['recId'] == rec_id) & (frames_df['driverId'] == driver_id)]
    if group.empty:
        print(f"No data found for Recording {rec_id}, Driver {driver_id}")
        return None
    
    group = group.sort_values(by='frame').reset_index(drop=True)
    params = params_df[(params_df['recId'] == rec_id) & (params_df['driverId'] == driver_id)]
    
    if params.empty:
        print(f"No parameters found for Recording {rec_id}, Driver {driver_id}")
        return None
    
    T, a, b, v0 = params.iloc[0][['T', 'a', 'b', 'v0']]

    # Simulation arrays
    sim_positions = [0.0]
    sim_velocity = group.loc[0, 'v']
    sim_accelerations = [calculate_idm_acc(sim_velocity, v0, T, 
                        max(group.loc[0, 's'], 0.01), s0, a, b, 
                        group.loc[0, 'delta_v'])] 
    actual_positions = [0.0]
    actual_velocity = group.loc[0, 'v']
    actual_accelerations = [group.loc[0, 'XAcceleration']]
    frames = [group.loc[0, 'frame']]

    for i in range(1, len(group)):
        row = group.loc[i]
        frames.append(row['frame'])
        s = max(row['s'], 0.01)
        delta_v = row['delta_v']

        # calc IDM acceleration
        v_idm = calculate_idm_acc(sim_velocity, v0, T, s, s0, a, b, delta_v)
        
        # SIMULATED
        sim_velocity += v_idm * dt
        sim_positions.append(sim_positions[-1] + sim_velocity * dt)
        sim_accelerations.append(v_idm)

        # ACTUAL XACC
        actual_velocity += row['XAcceleration'] * dt
        actual_positions.append(actual_positions[-1] + actual_velocity * dt)
        actual_accelerations.append(row['XAcceleration'])

    # Compute errors
   # ===== Corrected Error Calculation =====
    sim_positions = np.array(sim_positions)
    actual_positions = np.array(actual_positions)
    
    # Filter out invalid positions
    valid_mask = np.isfinite(sim_positions) & np.isfinite(actual_positions)
    valid_sim_pos = sim_positions[valid_mask]
    valid_actual_pos = actual_positions[valid_mask]
    
    if len(valid_sim_pos) < 2:
        print(f"Insufficient valid frames for error calculation: {len(valid_sim_pos)}")
        return None
    
    # Calculate RMSE
    squared_errors = (valid_sim_pos - valid_actual_pos)**2
    rmse = np.sqrt(np.mean(squared_errors))
    
    # Calculate average position (no artificial offset)
    avg_position = np.mean(np.abs(valid_actual_pos))
    error_percent = 0.0 if avg_position < 1e-5 else 100 * rmse / avg_position
    
    if not np.isfinite(error_percent):
        print(f"Invalid error percentage calculated: {error_percent}")
        return None

    # Create and save plot
    plt.figure(figsize=(10, 6))
    plt.plot(frames, sim_positions, 'b-', label='IDM Position', linewidth=1.5)
    plt.plot(frames, actual_positions, 'r--', label='Actual Position', linewidth=1.5)
    
    plt.annotate(f'Error: {error_percent:.1f}%',
                xy=(0.05, 0.9), xycoords='axes fraction',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.annotate(f'a={a:.3f}, b={b:.3f}\nT={T:.3f}, v0={v0:.3f}',
                 xy=(0.05, 0.75), xycoords='axes fraction',
                 fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title(f'Recording {rec_id} | Driver {driver_id}')
    plt.xlabel('Frame')
    plt.ylabel('Position (m)')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('graphs', exist_ok=True)
    plot_path = f'graphs/position_comparison_rec_{rec_id}_driver_{driver_id}.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Print detailed results
    print("\n=== Detailed Analysis for Specific Driver ===")
    print(f"Recording ID: {rec_id}")
    print(f"Driver ID: {driver_id}")
    print(f"Number of frames: {len(group)}")
    print("\nIDM Parameters:")
    print(f"Time headway (T): {T:.3f} s")
    print(f"Acceleration (a): {a:.3f} m/s²")
    print(f"Deceleration (b): {b:.3f} m/s²")
    print(f"Desired speed (v0): {v0:.3f} m/s")
    print("\nPerformance Metrics:")
    print(f"RMSE: {rmse:.3f} m")
    print(f"Error percentage: {error_percent:.1f}%")
    print(f"\nPlot saved to: {plot_path}")

    return {
        'recId': rec_id,
        'driverId': driver_id,
        'rmse': rmse,
        'error_percent': error_percent,
        'T': T,
        'a': a,
        'b': b,
        'v0': v0,
        'plot_path': plot_path
    }



# === Main Analysis ===
if __name__ == "__main__":
    TARGET_REC_ID = 1 
    TARGET_DRIVER_ID = 168 
    
    analyze_specific_driver(TARGET_REC_ID, TARGET_DRIVER_ID)
    
    # # === Run for all drivers and summarize ===
    # results = []
    # for (rec_id, driver_id) in params_df[['recId', 'driverId']].drop_duplicates().itertuples(index=False):
    #     res = analyze_driver(rec_id, driver_id)
    #     if res is not None:
    #         results.append(res)

    # # Convert to DataFrame
    # results_df = pd.DataFrame(results)

    # # Filter out high-error drivers
    # valid_results = results_df[results_df['error_percent'] <= 100]

    # mean_rmse = valid_results['rmse'].mean()
    # mean_error_percent = valid_results['error_percent'].mean()
