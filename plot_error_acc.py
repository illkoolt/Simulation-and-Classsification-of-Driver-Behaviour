import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
frames_df = pd.read_csv('idm_output_driver_frames.csv')

# Compute mean IDM acceleration
frames_df['mean_idm'] = frames_df[['idm_ga', 'idm_mle', 'idm_de']].mean(axis=1)


# === Settings === #
PRINT_PLOT = True
TARGET_REC_ID = 1
TARGET_DRIVER_ID = 228
SAVE_FOLDER = 'graphs'  # Folder to save plots
all_errors = []  # To store all errors for analysis

# Create directory if it doesn't exist
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Check if the specific driver exists in the recording
rec_df = frames_df[(frames_df['recId'] == TARGET_REC_ID) & 
                   (frames_df['driverId'] == TARGET_DRIVER_ID) & PRINT_PLOT]


# Create figure for this specific driver
plt.figure(figsize=(10, 6))

driver_df = rec_df.sort_values('frame')
x = driver_df['frame']
x_acc = driver_df['XAcceleration']
mean_idm_acc = driver_df['mean_idm']

error = np.abs(x_acc - mean_idm_acc)
mean_error = error.mean()
all_errors.append(mean_error)

# Plot for this driver
plt.plot(x, x_acc, 'r--', label='XAcceleration (true)', linewidth=1.5)
plt.plot(x, mean_idm_acc, 'b-', label='Mean IDM Acceleration', linewidth=1.5)

# Add annotations
plt.annotate(f'Mean Error: {mean_error:.3f} m/s²',
            xy=(0.05, 0.9), xycoords='axes fraction',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.title(f'Recording {TARGET_REC_ID} | Driver {TARGET_DRIVER_ID}')
plt.xlabel('Frame')
plt.ylabel('Acceleration (m/s²)')
plt.ylim(-6, 6)
plt.grid(True)
plt.legend(loc='lower right')

plt.tight_layout()

# Save the plot
plt.savefig(f'{SAVE_FOLDER}/acceleration_comparison_rec_{TARGET_REC_ID}_driver_{TARGET_DRIVER_ID}.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics if needed

print("\n=== Acceleration Error Analysis ===")
print(f"Processed {len(all_errors)} drivers")
print(f"Mean Error: {np.mean(all_errors):.3f} m/s²")
print(f"Max Error: {np.max(all_errors):.3f} m/s²")
print(f"Min Error: {np.min(all_errors):.3f} m/s²")
print(f"\nPlots saved to '{SAVE_FOLDER}' folder")
