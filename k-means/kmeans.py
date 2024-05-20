import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
iris_df = pd.read_csv("/content/data/Iris.csv")
X = iris_df.iloc[:, 1:-1].values  # Features
y = iris_df.iloc[:, 1:-1].values   # Target

# Number of clusters
k = 3

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Function to initialize centroids randomly
def initialize_centroids(data, k):
    centroids = []
    for _ in range(k):
        centroid = data[np.random.choice(range(len(data)))]
        centroids.append(centroid)
    return np.array(centroids)

# Function to assign each data point to the nearest centroid
def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

# Function to update centroids based on the mean of data points in each cluster
def update_centroids(data, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = [data[j] for j in range(len(data)) if clusters[j] == i]
        new_centroid = np.mean(cluster_points, axis=0)
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

# K-Means algorithm
def k_means(data, k):
    centroids = initialize_centroids(data, k)
    iter = 0
    converged = False
    sse = 0

    while not converged:
        iter += 1
        old_centroids = centroids.copy()

        # Assign data points to the nearest centroid
        clusters = assign_clusters(data, centroids)

        # Update centroids based on the mean of data points in each cluster
        centroids = update_centroids(data, clusters, k)

        # Check for convergence
        converged = np.array_equal(old_centroids, centroids)

    # Calculate SSE
    for i in range(k):
        cluster_points = data[clusters == i]
        sse += np.sum((cluster_points - centroids[i])**2)

    return clusters, centroids, sse

# Run K-Means clustering
kmeans_clusters, kmeans_centroids, kmeans_sse = k_means(X, k)

print(kmeans_centroids)

# Plot K-Means clusters
colors = ['r', 'g', 'b']
plt.figure(figsize=(8, 6))
for i in range(k):
    cluster_points = X[kmeans_clusters == i]
    _, counts = np.unique(kmeans_clusters, return_counts=True)
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i+1} - { counts[i] }')
    plt.scatter(kmeans_centroids[i, 0], kmeans_centroids[i, 1], marker='x', color='black', s=150)
plt.title('K-Means Clustering')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()
print("K-Means SSE:", kmeans_sse)