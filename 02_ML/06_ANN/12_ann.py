# ANN = Artificial Neural Network (Bio-Inspired)
# PERCETTRONE: insieme di input x_i (di solito normalizzati), di pesi w_i, un output y
# y=ɸ(z), z = sum_i(w_i*x_i)
# Funzione attivazione
#   Step Function: ɸ(z)=1 se z>=0
#   Sigmoid Funct: ɸ(z) = 1/(1+e^-z)
#   ReLU (Rectified Linear Unit): ɸ(z) = max(0, z)
#   TanH: ɸ(z) = (1-e^-2z)/(1+e^-2z)
# MLP: MultiLayer Perceptron -> idea simile a medoti Ensamble
#   Sigmoid Funct: ɸ(z) = 1/(1+e^-z)                [usate per output layer]
#   ReLU (Rectified Linear Unit): ɸ(z) = max(0, z)  [usata per hidden layers]
#   
#   Back propagation: Variazione pesi input in base al "Gradient Descent"
# example: https://playground.tensorflow.org


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

################ CANCER PREDICTION ##################àààà
cancer = load_breast_cancer() 
print(cancer['data'].shape) # 569 esempi, 30 features

X = cancer['data']
y = cancer['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30),verbose=True)
mlp.fit(X_train,Y_train)

Y_pred_train = mlp.predict(X_train)
Y_pred = mlp.predict(X_test)

print(f"ACCURACY: TRAIN={accuracy_score(Y_train, Y_pred_train)} TEST={accuracy_score(Y_test, Y_pred)}")

############# DIGITS RECOGNITION ################
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

digits = load_digits()
X = digits.data
Y = digits.target

plt.imshow(X[40].reshape([8,8]), cmap="gray")
plt.title(f"Numero {Y[40]}")
plt.show(block=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30),verbose=True)
mlp.fit(X_train,Y_train)

Y_pred_train = mlp.predict(X_train)
Y_pred = mlp.predict(X_test)

print(f"ACCURACY: TRAIN={accuracy_score(Y_train, Y_pred_train)} TEST={accuracy_score(Y_test, Y_pred)}")

for i in range(0, len(X_test)):
    if(Y_test[i]!=Y_pred[i]):
        plt.imshow(X_test[i].reshape([8,8]), cmap="gray")
        plt.title(f"Numero {Y_test[i]} classificato come {Y_pred[i]}")
        plt.show(block=True)



# https://playground.tensorflow.org
# https://affinelayer.com/pixsrv/
# https://www.youtube.com/watch?v=GrP_aOSXt5U&feature=youtu.be
