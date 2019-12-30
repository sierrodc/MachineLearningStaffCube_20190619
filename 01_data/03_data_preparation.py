import pandas as pd
import numpy as np

from sklearn.datasets import load_boston

boston = load_boston()
# boston.feature_names = names of the data columns
# boston.data = array of array of data
#boston.tar

print(boston.target_names)

X = boston.data
Y = boston.target

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

print(boston.feature_names)

boston_df = pd.DataFrame(
    data=np.c_[boston.data, boston.target],
    columns=np.append(boston.feature_names, 'TARGET')
)
boston_test_df = boston_df.sample(frac=0.3)
boston_train_df = boston_df.drop(boston_test_df.index)