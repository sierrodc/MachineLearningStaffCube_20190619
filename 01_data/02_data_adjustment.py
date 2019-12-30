import pandas as pd
import numpy as np

##############################################
## 01: from labels to numbers 
##############################################
data = pd.read_csv("data/shirts.csv", index_col=0)

size_mapping = {"S": 0, "M": 1, "L": 2, "XL": 3}

shirts_01 = data.copy()
shirts_01["taglia"] = shirts_01["taglia"].map(size_mapping) # apply map
shirts = pd.get_dummies(shirts_01, columns=['colore']) # create columns for each label instance

shirts_02 = data.copy().values
fmap = np.vectorize(lambda t: size_mapping[t]) # create a function that take array as input
shirts_02[:,0] = fmap(shirts_02[:,0])

from sklearn.preprocessing import LabelEncoder # text-column to number
from sklearn.preprocessing import OneHotEncoder # text-column to multiple column
le = LabelEncoder()
ohe = OneHotEncoder(categorical_features=[1])
shirts_02[:, 1] = le.fit_transform(shirts_02[:, 1]) #label to number
shirts_02_sparse = ohe.fit_transform(shirts_02)
shirts_02 = shirts_02_sparse.toarray()

#print(shirts_02[:5])

##############################################
## 02: handle null values = drop or fill with mean
##############################################
iris_na = pd.read_csv("data/iris_nan.csv")
X = iris_na.drop("class", axis=1).values
# null values removed from table
iris_na.dropna() #remove rows with na
iris_na.dropna(axis=1) #remove cols with na
# null values filled with mean/mode/median
mean = iris_na.mean()
isis_filled = iris_na.fillna(mean)

from sklearn.preprocessing import Imputer
imp = Imputer(strategy="mean", axis=0, missing_values="NaN") #median or most_frequent
X_imp =imp.fit_transform(X)

##############################################
## 03: normalization - standardization
##  - Scale = cambiare il range dei valori. Di solito tra [0 e 1]
##  - Standardize = cambiare il valore in modo che la deviazione standard è uguale a 1 (simile a N)
##  - WHY: https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
##############################################
wines = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
                        usecols=[0,1,7],
                        names=['classe', 'alcol', 'flavonoidi'])

Y = wines['classe'].values
X = wines.drop('classe', axis=1).values
features = ['alcol', 'flavonoidi']

#normalizzazione [0-1]
wines_norm = wines.copy()
to_norm = wines[features]
wines_norm[features] = (to_norm - to_norm.min())/(to_norm.max() - to_norm.min())
# equals to:
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_norm = X.copy()
X_norm = mms.fit_transform(X_norm)

#standardizzazione [-1,1]
wines_std = wines.copy()
to_std = wines_std[features]
wines_std[features] = (to_std - to_std.mean())/to_std.std()
# equals to:
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_std = X.copy()
X_std = ss.fit_transform(X_std)

print('')