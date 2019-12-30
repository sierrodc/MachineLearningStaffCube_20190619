# Data Leakage = se uso test-set per scegliere un modello/ottimizzarlo -> overfitting su test set!!!

# Soluzioni:
#   dataset diviso in:
#       -> training set
#       -> validation set -> scegliere/ottimizzare il modello
#       -> test set -> ultimo step per calcolare score
#   k-fold cross validation: divido dataset in k-fold
#       -> utilizza primo fold per test, gli altri per training --> modello 1 --> errore 1
#       -> utilizzo secondo fold per test, gli altri per training --> modello 2 ---> errore 2
#       -> ... --> modello k --> errore k
#       errore = avg(e1, e2, ... ek)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    names=["sepal length", "sepal width", "petal length", "petal width", "class"])

X = iris.drop("class", axis=1).values
Y = iris["class"].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

# KFold standard
from sklearn.model_selection import KFold
lr = LogisticRegression()
k = 10
kfold = KFold(n_splits=k, random_state=1)
scores = [] #k scores

for k, (train, test) in enumerate(kfold.split(X_train)):
    lr.fit(X_train[train], Y_train[train])
    score = lr.score(X_train[test], Y_train[test])
    scores.append(score)
    print(f"Fold: {k}, Score: {score}")

print(f"Total validation accuracy: {np.array(scores).mean()}")

# StratifiedKFold -> assicura presenza delle classi
from sklearn.model_selection import StratifiedKFold
lr = LogisticRegression()
k = 10
skfold = KFold(n_splits=k, random_state=1)
scores = [] #k scores

for k, (train, test) in enumerate(skfold.split(X_train, Y_train)):
    lr.fit(X_train[train], Y_train[train])
    score = lr.score(X_train[test], Y_train[test])
    scores.append(score)
    print(f"Fold: {k}, Score: {score}")

print(f"Total validation accuracy: {np.array(scores).mean()}")


# kfold
from sklearn.model_selection import cross_val_score
lr = LogisticRegression()
score = cross_val_score(lr, X_train, Y_train, cv=10)