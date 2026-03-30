
import numpy as np


def step(z):
    # Heaviside step function: returns 1 if z >= 0, else 0.
    return (z >= 0).astype(int)


class Perceptron:
    def __init__(self, n_features, lr=0.1, seed=None):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.w = rng.standard_normal(n_features) * 0.01
        self.b = 0.0
        self.errors = []

    def forward(self, x):
        return step(x @ self.w + self.b)

    def _update(self, x, y_true, y_pred):
        # Apply Rosenblatt's correction rule if prediction is wrong.
        delta = (y_true - y_pred)  # +1, -1, or 0
        self.w = self.w + self.lr * delta * x
        self.b = self.b + self.lr * delta
        return int(y_pred != y_true)

    def _train_one_epoch(self, X, y):
        # Run one pass over the dataset and return the number of errors.
        epoch_error = 0
        for i in range(len(X)):
            y_pred = self.forward(X[i])
            epoch_error += self._update(X[i], y[i], y_pred)
        self.errors.append(epoch_error)
        return epoch_error

    def train(self, X, y, epochs=100):
        # Train until convergence or max epochs is reached.
        for epoch in range(epochs):
            epoch_errors = self._train_one_epoch(X, y)
            if epoch_errors == 0:
                break
        return self.errors

    def predict(self, X):
        return self.forward(X)