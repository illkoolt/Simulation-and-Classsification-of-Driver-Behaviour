import pandas as pd

# Read the input files
frames_df = pd.read_csv('idm_output_driver_frames.csv')
params_df = pd.read_csv('idm_mean_parameters.csv')

# Group frames by recId and driverId to check for constant precedingId
def has_constant_preceding(group):
    # Get unique precedingId values (excluding possible NaN)
    unique_preceding = group['precedingId'].dropna().unique()
    return len(unique_preceding) == 1

# Find groups (driver trips) with constant precedingId
constant_preceding_drivers = frames_df.groupby(['recId', 'driverId']).apply(has_constant_preceding)
constant_preceding_drivers = constant_preceding_drivers[constant_preceding_drivers].index.tolist()

# Create a DataFrame with recId and driverId for filtering
constant_drivers_df = pd.DataFrame(constant_preceding_drivers, columns=['recId', 'driverId'])

# Rename columns to match params_df (which uses rec_id instead of recId)
constant_drivers_df = constant_drivers_df.rename(columns={
    'recId': 'rec_id',
    'driverId': 'driver_id'
})

# Filter the parameters DataFrame to only include drivers with constant precedingId
filtered_params = params_df.merge(constant_drivers_df, on=['rec_id', 'driver_id'])

# Save the filtered parameters to a new CSV file
filtered_params.to_csv('idm_mean_no_linechange.csv', index=False)

# Print the total number of drivers with no lane change
total_drivers_no_linechange = len(constant_preceding_drivers)
print(f"Total drivers with no line change: {total_drivers_no_linechange}")