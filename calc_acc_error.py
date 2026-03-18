import pandas as pd
import numpy as np
from collections import defaultdict

def calculate_driver_errors(frames_df):
    """Calculate acceleration errors and identify high-error cases"""
    algorithms = ['idm_ga', 'idm_mle', 'idm_de']
    driver_errors = {algo: defaultdict(list) for algo in algorithms}
    all_errors = {algo: [] for algo in algorithms}
    high_error_cases = {algo: [] for algo in algorithms}  # New: Store high-error cases
    
    # Group by recording and driver
    for (rec_id, driver_id), group in frames_df.groupby(['recId', 'driverId']):
        group = group.sort_values('frame')
        x_acc = group['XAcceleration'].values
        
        for algo in algorithms:
            algo_acc = group[algo].values
            
            try:
                # Calculate absolute errors
                errors = np.abs(x_acc - algo_acc)
                valid_mask = np.isfinite(errors)
                valid_errors = errors[valid_mask]
                
                if len(valid_errors) == 0:
                    continue
                
                mean_error = np.mean(valid_errors)
                
                # Store results
                driver_errors[algo][rec_id].append((driver_id, mean_error))
                all_errors[algo].append(mean_error)
                
                # Identify high-error cases (>60 m/s²)
                if mean_error > 60:
                    high_error_cases[algo].append({
                        'recId': rec_id,
                        'driverId': driver_id,
                        'mean_error': mean_error,
                        'num_frames': len(group),
                        'max_error': np.max(valid_errors)
                    })
                
            except Exception as e:
                print(f"Error processing {algo} for driver {driver_id}: {str(e)}")
                continue
    
    return driver_errors, all_errors, high_error_cases  # Now returns 3 items

def print_error_report(driver_errors, all_errors, high_error_cases):
    """Print comprehensive error report with high-error cases"""
    algorithms = ['idm_ga', 'idm_mle', 'idm_de']
    algo_names = {
        'idm_ga': 'Genetic Algorithm',
        'idm_mle': 'Maximum Likelihood Estimation',
        'idm_de': 'Differential Evolution'
    }
    
    for algo in algorithms:
        current_errors = np.array(all_errors[algo])
        
        print(f"\n=== {algo_names[algo]} Error Report ===")
        print(f"Total drivers analyzed: {len(current_errors)}")
        print(f"Mean error: {np.mean(current_errors):.3f} m/s²")
        print(f"Median error: {np.median(current_errors):.3f} m/s²")
        print(f"Minimum error: {np.min(current_errors):.3f} m/s²")
        print(f"Maximum error: {np.max(current_errors):.3f} m/s²")
        print(f"Standard deviation: {np.std(current_errors):.3f} m/s²")
        
        # Print high-error cases
        if high_error_cases[algo]:
            print(f"\nHigh-error drivers (>60 m/s²): {len(high_error_cases[algo])}")
            for case in sorted(high_error_cases[algo], key=lambda x: -x['mean_error']):
                print(f"Rec {case['recId']}, Driver {case['driverId']}: "
                      f"Mean={case['mean_error']:.3f} m/s², "
                      f"Max={case['max_error']:.3f} m/s², "
                      f"Frames={case['num_frames']}")
        else:
            print("\nNo drivers with errors >60 m/s²")

def main():
    try:
        # Load data
        frames_df = pd.read_csv('idm_output_driver_frames.csv')
        
        # Validate required columns
        required_cols = ['recId', 'driverId', 'frame', 'XAcceleration', 
                        'idm_ga', 'idm_mle', 'idm_de']
        missing_cols = [col for col in required_cols if col not in frames_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
            
        # Calculate errors and get high-error cases
        driver_errors, all_errors, high_error_cases = calculate_driver_errors(frames_df)
        
        # Print report with high-error cases
        print_error_report(driver_errors, all_errors, high_error_cases)
        
        # Save all results including high-error cases
        results = []
        for algo in ['idm_ga', 'idm_mle', 'idm_de']:
            for rec_id, drivers in driver_errors[algo].items():
                for driver_id, error in drivers:
                    results.append({
                        'algorithm': algo.replace('idm_', '').upper(),
                        'recId': rec_id,
                        'driverId': driver_id,
                        'mean_error': error,
                        'is_high_error': error > 60
                    })
        
        # pd.DataFrame(results).to_csv('error_analysis_with_high_cases.csv', index=False)
        print("\nCalculations completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()