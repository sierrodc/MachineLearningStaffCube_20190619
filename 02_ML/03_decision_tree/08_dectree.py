# banche e gruppi assicurativi
# root: nodo radice
# node: contiene un attributo/feature
# leaf: decisione
# link: rappresenta una decisione
# deep: lunghezza massima albero (più è lungo, più è complesso = overfitting)

# few algorithms such as CART (uses Gini Index, faster) or ID3 (entropy & information gains)
# entropia: H  = sum[ -p(c)*log2(p(c)) ]
# infogain: IG = E_before - sum[ p(t)*H(t) ]

# https://medium.com/deep-math-machine-learning-ai/chapter-4-decision-trees-algorithms-b93975f7a1f1

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

titanic = pd.read_csv("data/titanic.csv")

print(titanic.info())
print(titanic.head())

titanic = titanic.drop("Name", axis=1) #remove useless column
titanic = pd.get_dummies(titanic) #add sex_male and sex_female

X = titanic.drop("Survived", axis=1).values
Y = titanic["Survived"].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)
print(X_train.shape)

from sklearn.tree import DecisionTreeClassifier
for md in range(1, 10):
    tree = DecisionTreeClassifier(criterion="gini", max_depth=md)
    tree.fit(X_train, Y_train)

    Y_pred_train = tree.predict(X_train)
    Y_pred = tree.predict(X_test)
    
    print(f"DEPTH: {md} ACCURACY: TRAIN={accuracy_score(Y_train, Y_pred_train)} TEST={accuracy_score(Y_test, Y_pred)}")

from sklearn.tree import export_graphviz
file = open("tree.dot", "w")
export_graphviz(tree, out_file=file,
         feature_names=titanic.columns.drop("Survived"),
         class_names=titanic["Survived"].apply(str)) #http://webgraphviz.com/
file.close()
