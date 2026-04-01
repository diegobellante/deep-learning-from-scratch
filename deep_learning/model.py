
import numpy as np


def step(z):
    # Heaviside step function: returns 1 if z >= 0, else 0.
    return (z >= 0).astype(int)

class BCE:
   # ── Loss function: Binary Cross-Entropy ──────
   # -(y·log(ŷ) + (1-y)·log(1-ŷ)
   def forward(self, y_true, y_pred):
       eps = 1e-9
       return np.mean(-y_true * np.log(y_pred + eps)- (1 - y_true) * np.log(1 - y_pred + eps))  # average BCE loss over the batch 

   def backward(self, y_true, y_pred):
        eps = 1e-9
        N   = len(y_true)
        da=(1/N) * (-(y_true / (y_pred + eps)) + (1 - y_true) / (1 - y_pred + eps))  # ∂L/∂ŷ o ∂L/∂a
        return da # (batch,)

    
class Sigmoid:
   def forward(self, z):
        self.a = 1 / (1 + np.exp(-z))
        return self.a
   def backward(self):
       return self.a*(1- self.a) # la derivada de sigma = sigma*(1-sigma)  . La forma es #(batch,1)

class Neuron:
    def __init__(self, n_features, activation_function, loss_function, lr=0.1, seed=None):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.w = rng.standard_normal(n_features) * 0.01  # (n_features,)
        self.b = 0.0
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.losses = []

    def forward(self, xb):
        return  self.activation_function.forward((xb @ self.w )+ self.b)
        
    def backward(self, da, xb):
        dz = da * self.activation_function.backward()   #(batch,)
        dw = xb.T @ dz / len(xb)              # weigth gradient (feature,)
        db = np.mean(dz, axis=0)         #bias gradient= batch mean  → shape (n_neurons,)
        self.w -= self.lr * dw   #(feature,)
        self.b -= self.lr * db
             
 
    def _train_one_epoch(self, X, y, batch_size=1):
        epoch_loss = 0
        for start in range(0, len(X), batch_size):
            xb = X[start:start+batch_size]  # x batch (batch, features)
            yb = y[start:start+batch_size]  # y batch (batch,)
            y_pred = self.forward(xb)
            loss = self.loss_function.forward(yb, y_pred)
            epoch_loss += loss
            # Backward
            da = self.loss_function.backward(yb,y_pred)
            self.backward(da,xb)
        
        self.losses.append(epoch_loss)    
        return epoch_loss

    def train(self, X, y, epochs=100, batch_size=1):
        for epoch in range(epochs):
            epoch_loss = self._train_one_epoch(X, y, batch_size)
        return self.losses

    def predict(self, X):
        return self.forward(X)
        
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