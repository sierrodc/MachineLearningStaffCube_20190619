import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import pandas as pd

X, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.5, random_state=0)
plt.scatter(X[:,0], X[:,1], s=50)
plt.show(block=True)

from scipy.cluster.hierarchy import linkage, dendrogram
link_matrix = linkage(X, method="ward") # single, complete, average, centroid...
pd.DataFrame(link_matrix) # col 0,1 => id cluster; col 2 => distance; col 3=#item per cluster
dendrogram(link_matrix)
plt.show(block=True)


from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3)
y = ac.fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show(block=True)