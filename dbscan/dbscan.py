import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Load student data from CSV file and remove duplicates
student_data = pd.read_csv("/content/data/students.csv")
student_data = student_data.drop_duplicates()

# Check for missing values in the dataset
student_data.isnull().sum()

# Select all columns (replace ":" with the column index range if needed)
student_data = student_data.iloc[:, :]

# Display the first few rows of the dataset
student_data.head()

# Select relevant features related to lifestyle choices
lifestyle_features = student_data[['NumberOffriends', 'basketball', 'football', 'baseball', 'sports']]

# Convert lifestyle features to numpy array for clustering
lifestyle_features = np.array(lifestyle_features) 

def DBSCAN_C(X, eps_range, min_samples_range):
    # Standardize features
    silhouette_scores = {}
    best_score_cluster_labels = {}

    for eps in eps_range:
        for min_samples in min_samples_range:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            cluster_labels = dbscan.fit_predict(X_scaled)
            # Compute silhouette score
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores[(eps, min_samples)] = silhouette_avg
            best_score_cluster_labels[eps] = (cluster_labels, dbscan.core_sample_indices_)

    best_params, best_score = max(silhouette_scores.items(), key=lambda x: x[1])
    print(f"Best parameters (eps, min_samples): {best_params}, Silhouette Score: {best_score:.2f}")
    cluster_labels, core_sample_indices = best_score_cluster_labels[best_params[0]]
    core_samples_mask = np.zeros_like(cluster_labels, dtype=bool)
    core_samples_mask[core_sample_indices] = True
    return silhouette_avg, cluster_labels, core_samples_mask

# Define parameter ranges for DBSCAN
eps_range = [0.2, 0.5, 1.0]  # Adjust the range as needed
min_samples_range = [3, 5, 7]  # Adjust the range as needed

# Perform DBSCAN clustering and get silhouette score, cluster labels, and core sample mask
score, cluster_labels, core_samples_mask = DBSCAN_C(lifestyle_features, eps_range, min_samples_range)

print(score, cluster_labels, len(core_samples_mask))

# Determine the number of clusters
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
unique_labels = set(cluster_labels)
colors = ['y', 'b', 'g', 'r']

# Plot clustered data points
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (cluster_labels == k)

    xy = lifestyle_features[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 2], 'o', markerfacecolor=col,
             markeredgecolor='k',
             markersize=6)
    
    xy = lifestyle_features[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 2], 'o', markerfacecolor=col,
             markeredgecolor='k',
             markersize=6)
    
plt.xlabel("Number of friend")
plt.ylabel("friend Playing football")
plt.title(f'DB