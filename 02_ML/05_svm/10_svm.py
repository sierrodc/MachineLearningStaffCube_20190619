# SVM = Support Vector Machine - Linear
#  - Trovare la retta che massimizza la distanza tra 2 classi
#  - Usa un sottoinsieme dei dati per definire la soluzione
#  - Non è influenzato dagli outliers
#  - Usa gli esempi "ambigui" per creare il modello -> definiscono i vettori di supporto

# https://towardsdatascience.com/understanding-support-vector-machine-part-1-lagrange-multipliers-5c24a52ffc5e

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    names=["sepal length", "sepal width", "petal length", "petal width", "class"])

print(iris.head())

X = iris.drop("class", axis=1).values
Y = iris["class"].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

X2_train = X_train[:, :2]
X2_test = X_test[:, :2]

from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X2_train, Y_train)

print(f"ACCURACY: TRAIN={svc.score(X2_train, Y_train)} TEST={svc.score(X2_test, Y_test)}")