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

    # KMeans clustering <= main call
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_norm)

    # countig drivers in each cluster to determine labels
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


    #========================================#
    #print mean values for each cluster
    param_names = ['T', 'a', 'b', 'v0_mps']
    for cluster_num in range(3):
        cluster_indices = np.where(clusters == cluster_num)
        cluster_data = X[cluster_indices]
        mean_values = cluster_data.mean(axis=0)
        label = cluster_labels[cluster_num]

        print(f"\n=== {label} (Cluster {cluster_num}) ===")
        for name, value in zip(param_names, mean_values):
            print(f"Mean {name}: {value:.3f}")


    #  driver classifications
    classification = {}
    for i, key in enumerate(driver_keys):
        classification[key] = cluster_labels[clusters[i]]

    # Plot 3D clusters
    label_color_map = {
        'Normal': 'g',
        'Cautious': 'b',
        'Aggressive': 'r'
    }
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for cluster_num in range(3):
        cluster_indices = np.where(clusters == cluster_num)
        count = len(cluster_indices[0])
        label = cluster_labels[cluster_num]
        color = label_color_map[label]
        
        ax.scatter(
            X[cluster_indices, 0],  # T
            X[cluster_indices, 1],  # a
            X[cluster_indices, 2],  # b
            c=color,
            label=f"{label}: {count}",
            s=50,
            alpha=0.6
        )

    ax.set_xlabel('Time Headway (T)')
    ax.set_ylabel('Acceleration (a)')
    ax.set_zlabel('Comfortable Deceleration (b)')
    ax.set_title('Driver Classification by IDM Parameters (Simple K-means)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('driver_classification_simple.png')
    plt.close()

    return classification

#==============================================#

def rule_based_classify(T, a, b):
    if T < 1.0 and a > 2.0 and b > 2.0:
        return "Aggressive"
    elif T > 2.5 and a < 1.5 and b < 2.0:
        return "Cautious"
    else:
        return "Normal"

#==============================================#


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
    


    rule_based_results = {}
    for _, row in df.iterrows():
        try:
            rec_id = int(row['rec_id'])
            driver_id = int(row['driver_id'])
            T = float(row['T'])
            a = float(row['a'])
            b = float(row['b'])
            v0 = float(row['v0'])
            v_avg = float(row['v_average']) if 'v_average' in row else None

            # Apply rule-based classification
            rule_label = rule_based_classify(T, a, b)
            rule_based_results[(rec_id, driver_id)] = rule_label

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


    print("\nRule-Based Classification Summary")
    print("=================================")
    from collections import Counter

    # Count occurrences
    rule_counts = Counter(rule_based_results.values())
    for style in ['Aggressive', 'Normal', 'Cautious']:
        count = rule_counts.get(style, 0)
        print(f"{style}: {count} drivers")



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

    print("Classification complete. Output written to driver_classification_summary.txt and driver_classification.png")