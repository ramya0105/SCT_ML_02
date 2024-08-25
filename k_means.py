import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

df = pd.read_csv('customer_purchase_history.csv')

print("Initial Data Sample:")
print(df.head())

features = df[['total_spent', 'number_of_purchases', 'average_purchase_value']]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_data)

tsne = TSNE(n_components=2, random_state=42)
tsne_data = tsne.fit_transform(scaled_data)

tsne_df = pd.DataFrame(data=tsne_data, columns=['TSNE1', 'TSNE2'])
tsne_df['cluster'] = df['cluster']
colors = ['red', 'green', 'brown', 'violet', 'blue']

plt.figure(figsize=(12, 8))
scatter = plt.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], c=tsne_df['cluster'], cmap=mcolors.ListedColormap(colors), edgecolor='k', s=100)
plt.title('K-Means Clustering of Customers with 5 Clusters (t-SNE)')
plt.xlabel('Scaled Annual Income')  
plt.ylabel('Scaled Spending Score') 

legend_labels = [f'Cluster {i}' for i in range(n_clusters)]
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, linestyle='') for i in range(n_clusters)]
plt.legend(handles, legend_labels, title='Cluster')

plt.colorbar(scatter, label='Cluster')
plt.show()

print("\nData with Clusters:")
print(df.head())
