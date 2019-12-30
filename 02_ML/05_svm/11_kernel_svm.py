# Kernel SVM = Support Vector Machine - Non lineare (i.e.: persone accerchiate)
# Idea = aumento la dimensione del set di partenza (da 2d a 3d). Trovo il piano che separa. Rimappo a 2d.
#   -> Dispendioso!
# Kernel => metrica di somiglianza tra 2 tipi
#   Kernel Gaussiano - Radial Basis - RBF: K(x, l)  = e ^ -(||x-l||^2 / 2s^2)
#       s = sigma = più + piccolo, più i punti più distanti vanno a zero prima
#       l = landmark (uno o più di uno, si può fare una combinazione lineare)
#   Kernel Lienare : K(x,l) = <x,l>
#   Kernel sigmoidale: K(x,l) = tanh(1/2s^2<x,l> + r)
#   Kernel polinomiale: K(x,l) = (1/2s^2<x,l> + r)^d

# pic:
# https://towardsdatascience.com/understanding-support-vector-machine-part-2-kernel-trick-mercers-theorem-e1e6848c6c4d


import numpy as np 
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_circles
import matplotlib.pyplot as plt

X, Y = make_circles(factor=0.5, noise=0.2, random_state=1) 

plt.scatter(X[:,0], X[:,1], c=Y)
plt.show(block=True)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)
from sklearn.svm import SVC
svc = SVC(kernel="linear", probability=True) #calcola anche probabilità di appartenenza, ma basso dataset
svc.fit(X_train, Y_train)
print(f"ACCURACY: TRAIN={svc.score(X_train, Y_train)} TEST={svc.score(X_test, Y_test)}")

svc = SVC(kernel="rbf", probability=True)
svc.fit(X_train, Y_train)
print(f"ACCURACY: TRAIN={svc.score(X_train, Y_train)} TEST={svc.score(X_test, Y_test)}")

svc = SVC(kernel="sigmoid", probability=True)
svc.fit(X_train, Y_train)
print(f"ACCURACY: TRAIN={svc.score(X_train, Y_train)} TEST={svc.score(X_test, Y_test)}")

svc = SVC(kernel="poly", probability=True)
svc.fit(X_train, Y_train)
print(f"ACCURACY: TRAIN={svc.score(X_train, Y_train)} TEST={svc.score(X_test, Y_test)}")