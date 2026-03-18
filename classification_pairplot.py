import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter


def classify_drivers(idm_params):
    # data for clustering (T, a, b, v0)
    driver_keys = list(idm_params.keys())
    driver_data = []
    for key in driver_keys:
        params = idm_params[key]
        driver_data.append([
            params['GA_params']['T'],
            params['GA_params']['a'],
            params['GA_params']['b'],
            params['GA_params']['v0_mps']
        ])

    X = np.array(driver_data)

    # normalize for clustering
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_norm)

    
    cluster_stats = []
    for cluster_num in range(3):
        cluster_indices = np.where(clusters == cluster_num)
        cluster_data = X[cluster_indices]
        mean_values = cluster_data.mean(axis=0)
        cluster_stats.append((cluster_num, mean_values))  # (cluster_id, [T, a, b, v0])


    cluster_stats.sort(key=lambda x: x[1][0])  # sort by mean T

    # Assign labels: 
    cluster_labels = {
        cluster_stats[0][0]: 'Cautious',
        cluster_stats[1][0]: 'Aggressive',
        cluster_stats[2][0]: 'Normal',
    }

    # Create color mapping based on cluster labels
    color_map = {
        'Normal': 'green',
        'Cautious': 'blue',
        'Aggressive': 'red'
    }

    # Assign driver classifications
    classification = {}
    for i, key in enumerate(driver_keys):
        classification[key] = cluster_labels[clusters[i]]

    # Plot 3D clusters (using first 3 parameters for visualization)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for cluster_num in range(3):
        cluster_indices = np.where(clusters == cluster_num)
        count = len(cluster_indices[0])
        label = cluster_labels[cluster_num]
        ax.scatter(
            X[cluster_indices, 0],  # T
            X[cluster_indices, 1],  # a
            X[cluster_indices, 2],  # b
            c=color_map[label],
            label=f"{label}: {count}",
            s=50,
            alpha=0.6
        )

    ax.set_xlabel('Time Headway (T)')
    ax.set_ylabel('Acceleration (a)')
    ax.set_zlabel('Comfortable Deceleration (b)')
    ax.set_title('Driver Classification by IDM Parameters (Simple K-means)\n(Visualized with first 3 parameters)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('driver_classification_simple.png')
    plt.close()

    # Create pair plots to visualize all parameter combinations
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    param_names = ['T', 'a', 'b', 'v0']
    
    for i in range(4):
        for j in range(4):
            if i == j:
                # Diagonal - show histograms
                axes[i,j].hist(X[:,i], bins=20, color='gray', alpha=0.7)
                axes[i,j].set_title(f'Distribution of {param_names[i]}')
            else:
                # Off-diagonal - show scatter plots
                for cluster_num in range(3):
                    cluster_indices = np.where(clusters == cluster_num)
                    label = cluster_labels[cluster_num]
                    axes[i,j].scatter(
                        X[cluster_indices, j],
                        X[cluster_indices, i],
                        c=color_map[label],
                        label=label if i == 0 and j == 1 else "",
                        s=30,
                        alpha=0.6
                    )
                axes[i,j].set_xlabel(param_names[j])
                axes[i,j].set_ylabel(param_names[i])
    
    # Add legend only once and ensure correct order
    handles, labels = axes[0,1].get_legend_handles_labels()
    # Reorder to Normal, Cautious, Aggressive
    order = [labels.index('Normal'), labels.index('Cautious'), labels.index('Aggressive')]
    handles = [handles[idx] for idx in order]
    labels = [labels[idx] for idx in order]
    fig.legend(handles, labels, loc='upper right')
    
    plt.suptitle('Pairwise Parameter Relationships with Driver Clusters', y=1.02)
    plt.tight_layout()
    plt.savefig('driver_classification_pairplot.png')
    plt.close()

    return classification


#============================= MAIN =================================#
if __name__ == "__main__":
    df = pd.read_csv("idm_mean_no_linechange.csv")
    print(f"Total rows (drivers): {len(df)}")

    idm_params = {}
    for _, row in df.iterrows():
        try:
            rec_id = int(row['rec_id'])
            driver_id = int(row['driver_id'])
            T = float(row['T'])
            a = float(row['a'])
            b = float(row['b'])
            v0 = float(row['v0'])
            v_avg = float(row['v_average']) if 'v_average' in row else None

            idm_params[(rec_id, driver_id)] = {
                'GA_params': {
                    'T': T,
                    'a': a,
                    'b': b,
                    'v0_mps': v0
                },
                'v_average': v_avg
            }
        except (ValueError, KeyError) as e:
            print(f"skip row due to error: {e}")
            continue

    if not idm_params:
        raise ValueError("No valid driver data found. Check your CSV file.")

    driver_classification = classify_drivers(idm_params)

    # Save classification summary
    with open('driver_classification_summary.txt', 'w') as f:
        f.write("Driver Classification Summary\n")
        f.write("===========================\n\n")
        for style in ['Aggressive', 'Normal', 'Cautious']:
            drivers = [
                f"{rec_id}:{driver_id}"
                for (rec_id, driver_id), cls in driver_classification.items()
                if cls == style
            ]
            f.write(f"{style} drivers ({len(drivers)}):\n")
            f.write(", ".join(sorted(drivers, key=lambda x: (int(x.split(':')[0]), int(x.split(':')[1])))))
            f.write("\n\n")

    print("Classification complete. Output written to:")
    print("- driver_classification_summary.txt")
    print("- driver_classification_simple.png (3D plot)")
    print("- driver_classification_pairplot.png (all parameter relationships)")