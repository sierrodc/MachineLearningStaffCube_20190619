import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=0)
plt.scatter(X[:,0], X[:,1], s=50)
plt.show(block=True)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y = kmeans.predict(X)
plt.scatter(X[:,0], X[:,1], c=y, s=25)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], c="red", s=150)
plt.show(block=True)

ssd = {}
for k in range(1,10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    ssd[k] = kmeans.inertia_

plt.plot(list(ssd.keys()), list(ssd.values()))
plt.xlabel("# clusters")
plt.ylabel("measure")
plt.show()
plt.show(block=True)
