import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
iris_df = pd.read_csv("/content/data/Iris.csv")
X = iris_df.iloc[:, 1:-1].values  # Features
iris_df.head()

#Number of cluster 
K = 3

# Function to initialize centroids randomly
def initialize_medoids(data, k):
  medoids = []
  for _ in range(k):
    medoid = data[np.random.choice(range(len(data)))]
    medoids.append(medoid)

  return medoids;

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def assign_clusters(data, medoids):
  clusters = []
  for point in data:
    distance = [euclidean_distance(point, medoid) for medoid in medoids]
    cluster = np.argmin(distance)
    clusters.append(cluster)
  return np.array(clusters)

def update_medoids(data, clusters, k):
  new_medoids = np.zeros((k, data.shape[1]))
  for i in range(k):
      cluster_points = data[clusters == i]
      distances_within_cluster = np.sum(np.sqrt(np.sum((cluster_points[:, np.newaxis] - cluster_points[np.newaxis, :]) ** 2, axis=2)), axis=1)
      new_medoids[i] = cluster_points[np.argmin(distances_within_cluster)]
  return new_medoids


def k_medoids(data, k):
  medoids = initialize_medoids(data, k)
  coverage = False
  itr = 0
  sse = 0
  while not coverage:
    itr += 1
    old_medoids = medoids.copy()
    clusters = assign_clusters(data, medoids)
    medoids = update_medoids(data, clusters, k)
    coverage = np.array_equal(old_medoids, medoids)

  for i in range(k):
        cluster_points = data[clusters == i]
        sse += np.sum((cluster_points - medoids[i])**2)

  return clusters, medoids, sse 

k_medoid_clusters, k_medoids, k_medoids_sse = k_medoids(X, K)
colors = ['r', 'g', 'b']
plt.figure(figsize=(8, 6))

print(k_medoids)

plt.scatter(X[:, 0], X[:, 1], c='r', label=f'Cluster')
for i in range(K):
  cluster_points = X[k_medoid_clusters == i]
  _, counts = np.unique(k_medoid_clusters, return_counts=True)
  plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i+1} - { counts[i] }')
  plt.scatter(k_medoids[i, 0], k_medoids[i, 1], marker='*', color='black', s=200)

plt.title('K-Medoids Clustering')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()
print("K - Medoids_sse SSE:", k_medoids_sse)
