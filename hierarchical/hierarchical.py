import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import seaborn as sns
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.metrics import silhouette_score

# Load student data from CSV file and remove duplicates
student_data = pd.read_csv("/content/data/students.csv")
student_data = student_data.drop_duplicates()
student_data.isnull().sum()

# Select all columns (replace ":" with the column index range if needed)
student_data = student_data.iloc[:, :]

# Select relevant features related to lifestyle choices
lifestyle_features = student_data[['music', 'shopping', 'clothes', 'hollister', 'abercrombie',
                                   'die', 'death', 'drunk', 'drugs']]

def aggromative_clustering(X, cluster):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Perform Agglomerative Clustering
    result = AgglomerativeClustering(n_clusters=cluster, linkage='ward')
    y_pred = result.fit_predict(X)
    cluster_labels = result.labels_
    return silhouette_score(X, y_pred), cluster_labels

def print_dendo(X):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Plot dendrogram
    link_matrix = linkage(X, method='ward')
    dendrogram(link_matrix)

n_clusters = 3
score, cluster_labels = aggromative_clustering(lifestyle_features, 3)

# Increase recursion limit to prevent potential RecursionError
sys.setrecursionlimit(len(lifestyle_features))
# Plot dendrogram
print_dendo(lifestyle_features)
plt.title('Dendrogram', fontsize=20)
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# Add cluster labels to the dataframe
student_data['Cluster'] = cluster_labels

# Plot each cluster's data
for cluster in range(n_clusters):
    cluster_data = student_data[student_data['Cluster'] == cluster]
    plt.scatter(cluster_data['shopping'], cluster_data['clothes'], label=f'Cluster {cluster} - {len(cluster_data)}')

plt.xlabel('Students who love shopping')
plt.ylabel('Clothes')
plt.title(f'Agglomerative Clustering with silhouette-score {score}')
plt.legend()
plt.show()
