import pandas as pd
import numpy as np

def calculate_gap_errors(csv_path):
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Calculate gap errors for each algorithm
    df['gap_error_ga'] = np.abs(df['s'] - df['s_star_ga'])
    df['gap_error_mle'] = np.abs(df['s'] - df['s_star_mle'])
    df['gap_error_de'] = np.abs(df['s'] - df['s_star_de'])
    
    # Group by driver to get per-driver statistics
    driver_stats = df.groupby(['recId', 'driverId']).agg({
        'gap_error_ga': 'mean',
        'gap_error_mle': 'mean',
        'gap_error_de': 'mean'
    }).reset_index()
    
    return driver_stats

def print_error_reports(driver_stats):
    algorithms = {
        'GA': 'Genetic Algorithm',
        'MLE': 'Maximum Likelihood Estimation',
        'DE': 'Differential Evolution'
    }
    
    for algo_prefix, algo_name in algorithms.items():
        error_col = f'gap_error_{algo_prefix.lower()}'
        errors = driver_stats[error_col]
        
        print(f"\n=== {algo_name} Gap Error Report ===")
        print(f"Total drivers analyzed: {len(driver_stats)}")
        print(f"Mean error: {errors.mean():.3f} m")
        print(f"Median error: {errors.median():.3f} m")
        print(f"Minimum error: {errors.min():.3f} m")
        print(f"Maximum error: {errors.max():.3f} m")
        print(f"Standard deviation: {errors.std():.3f} m")

def main():
    input_csv = 'idm_output_driver_frames.csv'
    
    # Calculate gap errors
    driver_stats = calculate_gap_errors(input_csv)
    
    # Print reports for each algorithm
    print_error_reports(driver_stats)
    
    # Save the results
    #driver_stats.to_csv('gap_error_report.csv', index=False)
    print("\nResults saved to 'gap_error_report.csv'")

if __name__ == "__main__":
    main()