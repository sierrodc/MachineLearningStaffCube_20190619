# Linear Discriminant Analysis = tiene in considerazione anche Y per miglior sottospazio

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    names=["sepal length", "sepal width", "petal length", "petal width", "class"])

X = iris.drop("class", axis=1).values
Y = iris["class"].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_train = pca.fit_transform(x_train)
pca_test = pca.transform(x_test)

plt.xlabel("PCA: 1st")
plt.ylabel("PCA: 2nd")
plt.scatter(pca_train[:, 0], pca_train[:, 1], c=y_train, edgecolors="black")
plt.scatter(pca_test[:, 0], pca_test[:, 1], c=y_test, edgecolors="black", alpha=0.5)
plt.show(block=True)


lr = LogisticRegression()
lr.fit(pca_train, y_train)
print(f"AS_TEST: {accuracy_score(y_train, lr.predict(pca_train))}, AS_TEST: {accuracy_score(y_test, lr.predict(pca_test))}")


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
lda_train = lda.fit_transform(x_train, y_train)
lda_test = lda.transform(x_test)

plt.xlabel("LDA: 1st")
plt.ylabel("LDA: 2nd")
plt.scatter(lda_train[:, 0], lda_train[:, 1], c=y_train, edgecolors="black")
plt.scatter(lda_test[:, 0], lda_test[:, 1], c=y_test, edgecolors="black", alpha=0.5)
plt.show(block=True)

lr = LogisticRegression()
lr.fit(lda_train, y_train)
print(f"AS_TEST: {accuracy_score(y_train, lr.predict(lda_train))}, AS_TEST: {accuracy_score(y_test, lr.predict(lda_test))}")

