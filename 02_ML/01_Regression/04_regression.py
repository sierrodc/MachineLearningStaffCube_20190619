##############################################
## 01: REGRESSIONE LINEARE SEMPLICE
##  y = f(x) = b + wx
## Residual Sum of Squares (RSS) = sum_i[(y_i - (b+wx_i))^2]             OTTIMO = 1
## Mean Squared error (MSE) = sum_i[(y_i - (b+wx_i))^2] / max(i)         MEDIA DEL QUADRATO DEGLI ERRORI, OTTIMO=0
##############################################

import pandas as pd
import numpy as np

#"https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
house = pd.read_csv("data/housing.data",
        sep="\s+", usecols=[5,13], names=["RM", "MEDV"]) #numero camere + valore medio case

X = house.drop("MEDV", axis=1).values
Y = house["MEDV"].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
print(f"1-var MSE: {mean_squared_error(Y_test, Y_pred)}")
print(f"1-var R2 SCORE: {r2_score(Y_test, Y_pred)}")
 # < 0.3 useless
 # < 0.5 bad
 # < 0.7 good
 # < 0.9 very good
 # < 1.0 excellent
 # = 1 overfitting?

import matplotlib.pyplot as plt
# print("Peso di RM: " + str(lr.coef_[0]))
# print("Bias: " + str(lr.intercept_))
plt.scatter(X_train, Y_train, c="green", edgecolors="white", label="Train set")
plt.scatter(X_test, Y_test, c="blue", edgecolors="white", label="Test set")
plt.xlabel("Avg number of rooms [RM]")
plt.ylabel("Price (k$) [MEDV]")
plt.legend(loc="upper left")
plt.plot(X_test, Y_pred, color="red", linewidth=3)
plt.show(block=True)

##############################################
## 02: Multiple linear regression
##  y = f(x1, x2, ... xn) = b + w1x1 + w2x2 + ... + wnxn
##############################################
boston = pd.read_csv("data/housing.data",
        sep="\s+", names=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PRATIO", "B", "LSTAT", "MEDV"])

print(boston.corr())
 # -1 => correlazione inversa (al crescere di K diminuisce)
 #  0 => non correlato
 # +1 => correlazione diretta (al crescere di K cresce)
import seaborn as sb
sb.heatmap(boston.corr(), xticklabels=boston.columns, yticklabels=boston.columns, annot=True, annot_kws={'size': 12})
plt.show(block=True)

cols = ["RM", "LSTAT", "PRATIO", "TAX", "INDUS", "MEDV"]
sb.heatmap(boston[cols].corr(), xticklabels = cols, yticklabels = cols, annot=True, annot_kws={'size': 12})
#sb.pairplot(boston[cols]) # 
plt.show(block=True)

# with previous things, I can say that MEDV is highly correlated with RM (~0.7) and LSTAT (~-0.7)
X = boston[["RM", "LSTAT"]].values
Y = boston["MEDV"].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)

print(f"2-var MSE: {mean_squared_error(Y_test, Y_pred)}")
print(f"2-var R2 SCORE: {r2_score(Y_test, Y_pred)}")

# with all variables
X = boston.drop("MEDV", axis=1).values
Y = boston["MEDV"].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)

print(f"n-var MSE: {mean_squared_error(Y_test, Y_pred)}")
print(f"n-var R2 SCORE: {r2_score(Y_test, Y_pred)}")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train) # calcola media e varianza e applica trasformazione
X_test = ss.transform(X_test) # applica trasformazione con media e varianza precedenti

lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)

print(f"n-var (std) MSE: {mean_squared_error(Y_test, Y_pred)}")
print(f"n-var (std) R2 SCORE: {r2_score(Y_test, Y_pred)}")

print(list(zip(boston.columns, lr.coef_)))


##############################################
## 02: Polinomial linear regression
##  y = f(x1, x2, ... xn) = b + w1x1 + w2x2 + ... + wnxn
##############################################
boston = pd.read_csv("data/housing.data",
        sep="\s+", names=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PRATIO", "B", "LSTAT", "MEDV"])

X = boston[["RM", "LSTAT"]].values
Y = boston["MEDV"].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
from sklearn.preprocessing import PolynomialFeatures
# transorm [ [a] , [b], [c]...] to [ [1,a,a^2...], [1,b,b^2...], [1,c,c^2...] ...]

for i in range(1, 11): #from 1 to 10
        pf = PolynomialFeatures(degree=i)
        X_train_poly = pf.fit_transform(X_train)
        X_test_poly = pf.transform(X_test)
        lr = LinearRegression()
        lr.fit(X_train_poly, Y_train)
        Y_pred = lr.predict(X_test_poly)
        print(f"1-var {i} degree MSE: {mean_squared_error(Y_test, Y_pred)} R2 SCORE: {r2_score(Y_test, Y_pred)}")


X = boston.drop("MEDV", axis=1).values
Y = boston["MEDV"].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
from sklearn.preprocessing import PolynomialFeatures
# transorm [ [a] , [b], [c]...] to [ [1,a,a^2...], [1,b,b^2...], [1,c,c^2...] ...]

for i in range(1, 11): #from 1 to 10
        pf = PolynomialFeatures(degree=i)
        X_train_poly = pf.fit_transform(X_train)
        X_test_poly = pf.transform(X_test)
        lr = LinearRegression()
        lr.fit(X_train_poly, Y_train)
        Y_pred = lr.predict(X_test_poly)
        print(f"1-var {i} degree MSE: {mean_squared_error(Y_test, Y_pred)} R2 SCORE: {r2_score(Y_test, Y_pred)}")