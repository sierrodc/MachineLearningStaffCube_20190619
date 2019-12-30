import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
        sep="\s+", names=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PRATIO", "B", "LSTAT", "MEDV"])

X = boston.drop('MEDV', axis=1).values
Y = boston['MEDV'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

pf = PolynomialFeatures(degree=2)
X_train_pf = pf.fit_transform(X_train)
X_test_pf = pf.transform(X_test)

ss = StandardScaler()
X_train_pf_s = ss.fit_transform(X_train_pf)
X_test_pf_s = ss.transform(X_test_pf)

ll = LinearRegression()
ll.fit(X_train_pf_s, Y_train)
Y_pred_train = ll.predict(X_train_pf_s) # predizione del training set!

print("VS TRAINING SET: MSE = {}, R2_SCORE = {}".format(mean_squared_error(Y_train, Y_pred_train), r2_score(Y_train, Y_pred_train)))

Y_pred_test = ll.predict(X_test_pf_s)
print("VS TEST SET: MSE = {}, R2_SCORE = {}".format(mean_squared_error(Y_test, Y_pred_test), r2_score(Y_test, Y_pred_test)))


###################
## REGOLARIZZAZIONE: penalizza pesi più grandi, riducendo varianza e aumentando il bias. Aggiunge fattore di costo basato solo su pesi
## L2: cost(W, b) + lambda*SUM_j=1_m[Wj^2] -> aggiunge somma dei quadrati dei pesi (weight decay)
## L1: cost(W, b) + lambda*SUM_j=1_m[ |Wj| ] -> aggiunge somma del valore assoluto dei pesi. Pesi minori portati a zero.
## lambda (parametro regolarizzazione): quando pesa questo fattore extra. se 0 = regressione normale. Usually 10^-4 < lambda < 10^1
###################

from sklearn.linear_model import Ridge #regolarizzazione L2
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train_pf_s, Y_train)
    Y_pred_train = model.predict(X_train_pf_s)
    Y_pred_test = model.predict(X_test_pf_s)
    print("RIDGE ALPHA={} MSE: {} vs {} ---- R2_SCORE: {} vs {}".format(
        alpha,
        mean_squared_error(Y_train, Y_pred_train),
        mean_squared_error(Y_test, Y_pred_test),
        r2_score(Y_train, Y_pred_train),
        r2_score(Y_test, Y_pred_test)))

from sklearn.linear_model import Lasso #regolarizzazione L1
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
for alpha in alphas:
    model = Lasso(alpha=alpha)
    model.fit(X_train_pf_s, Y_train)
    Y_pred_train = model.predict(X_train_pf_s)
    Y_pred_test = model.predict(X_test_pf_s)
    print("LASSO ALPHA={} MSE: {} vs {} ---- R2_SCORE: {} vs {}".format(
        alpha,
        mean_squared_error(Y_train, Y_pred_train),
        mean_squared_error(Y_test, Y_pred_test),
        r2_score(Y_train, Y_pred_train),
        r2_score(Y_test, Y_pred_test)))


from sklearn.linear_model import ElasticNet #regolarizzazione L1+L2
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
for alpha in alphas:
    model = ElasticNet(alpha=alpha, l1_ratio=0.5)
    model.fit(X_train_pf_s, Y_train)
    Y_pred_train = model.predict(X_train_pf_s)
    Y_pred_test = model.predict(X_test_pf_s)
    print("ELSASTICNET ALPHA={} MSE: {} vs {} ---- R2_SCORE: {} vs {}".format(
        alpha,
        mean_squared_error(Y_train, Y_pred_train),
        mean_squared_error(Y_test, Y_pred_test),
        r2_score(Y_train, Y_pred_train),
        r2_score(Y_test, Y_pred_test)))