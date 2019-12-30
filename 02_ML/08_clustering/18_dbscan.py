# PRO:
#   no numero clustering a priori!
#   no cluster circolari
#   resiste a outliers
# VS:
#   definire a priori
#       eps = distanza massima 2 osservazioni
#       minPoints = #minimo osservazioni per formare un cluster (rule >= #dimensioni + 1)

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)
plt.scatter(X[:,0], X[:,1])
plt.show(block=True)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=2)
y_km = km.fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=y_km)
plt.show(block=True)

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2, linkage="ward")
y_ac = ac.fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=y_ac)
plt.show(block=True)

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.25, min_samples=3)
y_db = db.fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=y_db)
plt.show(block=True)