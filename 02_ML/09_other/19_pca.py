import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
cols = ["label", "alcol", "acido malico", "cenere", "alcalinità cenere", "magnesio", "fenoli tot", "flavonoidi", "fenoli non-flavonoidi", "proantocianidine",
         "intensità colore", "tonalità", "OD280/OD315 vinidiluiti", "prolina"]

wines = pd.read_csv(url, names=cols)
print(wines.head())

X = wines.drop("label", axis=1).values
Y = wines["label"].values

ss = StandardScaler()
X = ss.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(X)

plt.xlabel("PCA: x[1]")
plt.ylabel("PCA: x[2]")
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=Y, edgecolors="black")
plt.show(block=True)


#------------

pca_all = PCA(n_components=None)
pca_all.fit(X)
#print(pca_all.explained_variance_ratio_)
plt.xlabel("# componenti principali")
plt.ylabel("Varianza")
plt.step(range(1,14), np.cumsum(pca_all.explained_variance_ratio_))
plt.bar(range(1,14), pca_all.explained_variance_ratio_)
plt.show(block=True)