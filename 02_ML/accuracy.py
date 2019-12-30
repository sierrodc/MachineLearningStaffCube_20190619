import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 200, 3]
print(f"normalized: {accuracy_score(y_true, y_pred)}") # 0.5
print(f"absolute  : {accuracy_score(y_true, y_pred, normalize=False)}") #2