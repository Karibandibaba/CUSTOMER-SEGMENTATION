import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('Mall1_Customers.csv')  



X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
title='Elbow Method'
plt.title(title)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.savefig(f'static/images/{title.lower()}.png',bbox_inches='tight')
plt.show()

# Based on the Elbow Method graph, choose the optimal number of clusters (e.g., 5)
optimal_clusters = 5

# Apply K-Means clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add the cluster labels back to the original DataFrame
data['Cluster'] = cluster_labels

# Visualize the clusters
plt.figure(figsize=(10, 6))
for cluster in range(optimal_clusters):
    plt.scatter(X_scaled[cluster_labels == cluster][:, 0], X_scaled[cluster_labels == cluster][:, 1], label=f'Cluster {cluster}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
title='Customer Clusters'
plt.title(title)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.savefig(f'static/images/{title.lower()}.png',bbox_inches='tight')
plt.show()

# Create a summary table of cluster characteristics
cluster_summary = data.groupby('Cluster').agg({
    'Annual Income (k$)': ['mean', 'std'],
    'Spending Score (1-100)': ['mean', 'std', 'count']
}).reset_index()

# Rename columns for clarity
cluster_summary.columns = ['Cluster', 'Avg Income (k$)', 'Income Std Dev', 'Avg Spending Score', 'Spending Score Std Dev', 'Count']

# Display the summary table
print(cluster_summary)

# Visualize the summary table as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(cluster_summary['Cluster'], cluster_summary['Count'], color='skyblue')
title='Cluster Size'
plt.title(title)
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.savefig(f'static/images/{title.lower()}.png',bbox_inches='tight')
plt.show()