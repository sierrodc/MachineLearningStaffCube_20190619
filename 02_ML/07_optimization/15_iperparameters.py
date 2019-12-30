import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time

iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    names=["sepal length", "sepal width", "petal length", "petal width", "class"])

X = iris.drop("class", axis=1).values
Y = iris["class"].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

svc = SVC()
params= {
    "kernel": ["linear", "rbf", "sigmoid", "poly"],
    "gamma": [0.1, 1, "auto"],
    "C": [1, 10, 100, 1000]
}
gs = GridSearchCV(svc, params, cv=10)
start_time = time.time()
gs.fit(X_train, Y_train)

print(f"found best iperparameters in {time.time()-start_time} seconds: {gs.best_params_}, achieving score={gs.best_score_}")

#.............
#.............
#.............
from sklearn.model_selection import RandomizedSearchCV
svc = SVC()
rs = RandomizedSearchCV(svc, params, cv=10)
start_time = time.time()
gs.fit(X_train, Y_train)

print(f"found best iperparameters in {time.time()-start_time} seconds: {gs.best_params_}, achieving score={gs.best_score_}")
