# kNN è un modello lazy: non apprende dai dati, ma li memorizza
# Necessita il concetto di "distanza"
#  euclidea = (x1-x2)^2 + (y1-y2)^2
#  manhattan = |x1-x2| + |y1-y2|
#  minkowski (|x1-x2|^p + |y1-y2|^p)^1/p
# k = migliori elementi vicini (di solito ~5)

# pro: semplice e si adatta ai nuovi dati
# vs : quale metrica e dispendioso per set 

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

digits = load_digits()
X = digits.data
Y = digits.target

plt.imshow(X[40].reshape([8,8]), cmap="gray")
plt.show(block=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

Ks = [1,2,3,4,5,7,10,12,15,20]

for K in Ks:
    print("K="+str(K))
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,Y_train)
    
    y_pred_train = knn.predict(X_train) # return the predictions
    y_prob_train = knn.predict_proba(X_train) # return the probability of the predictions
    
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)
    print(f"ACCURACY: TRAIN={accuracy_score(Y_train, y_pred_train)} TEST={accuracy_score(Y_test, y_pred)}")
    print(f"LOG LOSS: TRAIN={log_loss(Y_train, y_prob_train)} TEST={log_loss(Y_test, y_prob)}")

found = False
for i in range(0, len(X_test)):
    if(Y_test[i]!=y_pred[i]):
        print(f"Numero {Y_test[i]} classificato come {y_pred[i]}")
        if(found == False):
            plt.imshow(X_test[i].reshape([8,8]), cmap="gray")
            plt.show(block=True)
        found = True
