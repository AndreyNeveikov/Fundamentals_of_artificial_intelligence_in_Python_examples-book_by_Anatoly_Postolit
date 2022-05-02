# Listing 6.21
from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.1, random_state=0, max_iter=40)
ppn.fit(X_train_std, y_train)
