import numpy as np # package for scientific computing with Python
import pandas as pd # easy-to-use data structures and data analysis tools

iris = pd.read_csv("data/iris.csv")

print(iris.head(10)) #first 10 rows
print(iris.tail(10)) #last 10 rows

print(iris.columns) #the columns
print(iris.info()) #info about the dataframe

y = iris['species'] # serie
#print(type(y))

X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
X = iris.drop('species', axis=1) # axis=1 => all rows remove 'species'

iris_sampled = iris.sample(frac=0.5)
iris_sampled.loc[3] # return row of index = 3 (ps: set_index('species'))
iris_sampled.iloc[32] # return row with first column = 32
#print(iris_sampled.iloc[0:10, 0:3]) #first 10 rows (from idx=0 to 10 excluded), column 0,1,2
#print(iris_sampled.iloc[0:10:2, 1:3]) # rows 0,2,4,6,8
#print(iris_sampled.describe()) # show iris.count(), iris.mean(), iris.var(), iris.min()
print(iris_sampled['species'].unique())

mask = iris['petal_length'] > iris['petal_length'].mean()
iris_long_petal = iris[mask]
#print(iris_long_petal.describe())

sorted = iris.sort_values('petal_length');

#print(iris.groupby('species').mean()) # species on rows, compute mean


import matplotlib.pyplot as plt
iris.plot(x='sepal_length', y='sepal_width', kind='scatter')
plt.show(block=True)