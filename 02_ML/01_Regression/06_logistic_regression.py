###
### o(z) = 1 / (1+e^-z)
### L(W) = P(Y|X,W) = LIKELIHOOD
###


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

breast_cancer = pd.read_csv("data/wdbc.data", names=[
    "id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean",
    "concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se",
    "area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst",
    "concave points_worst","symmetry_worst","fractal_dimension_worst"])

#print(breast_cancer.info())
#print(breast_cancer["diagnosis"].unique()) #M=maligno, B=benigno
X = breast_cancer[["radius_se", "concave points_worst"]].values #errore standard raggio, punti concavi
Y = breast_cancer["diagnosis"].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)
#M=>1, B=>0

#standardizziamo
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2', C=1) #regolarizzazione L2
lr.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss #log likelihood
Y_pred = lr.predict(X_test)
Y_pred_proba = lr.predict_proba(X_test) # probabilità correttezza della predizione

print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}, LOG_LOSS: {log_loss(Y_test, Y_pred_proba)}")

def showBounds(model, X, Y, labels=["Negativo","Positivo"]):
    
    h = .02 

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    X_m = X[Y==1]
    X_b = X[Y==0]
    plt.scatter(X_b[:, 0], X_b[:, 1], c="green",  edgecolor='white', label=labels[0])
    plt.scatter(X_m[:, 0], X_m[:, 1], c="red",  edgecolor='white', label=labels[1])
    plt.legend()
    plt.show(block=True)

showBounds(lr, X_train, Y_train, labels=["Benigno","Maligno"])

######
## Classificazione multiclasse con OneVsAll
######

from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
Y = digits.target

plt.imshow(X[40].reshape([8,8]), cmap="gray")
plt.show(block=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)
Y_pred_proba = lr.predict_proba(X_test) # probabilità correttezza della predizione

print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}, LOG_LOSS: {log_loss(Y_test, Y_pred_proba)}")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
import seaborn as sns
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True,  linewidths=0.5, square=True)
plt.ylabel('classe corretta')
plt.xlabel('classe predetta')
plt.show(block=True)