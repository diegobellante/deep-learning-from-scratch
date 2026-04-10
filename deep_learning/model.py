
import numpy as np


def step(z):
    # Heaviside step function: returns 1 if z >= 0, else 0.
    return (z >= 0).astype(int)

class ReLU:
   def forward(self, z):
        self.z = z
        return np.maximum(0,z)
   def backward(self):
       return (self.z > 0).astype(int) #  da · ∂a/∂z   Shape: #(batch,1)
       
class BCE:
   # ── Loss function: Binary Cross-Entropy ──────
   # -(y·log(ŷ) + (1-y)·log(1-ŷ)
   def forward(self, y_true, y_pred):
       eps = 1e-9
       return np.mean(-y_true * np.log(y_pred + eps)- (1 - y_true) * np.log(1 - y_pred + eps))  # average BCE loss over the batch 

   def backward(self, y_true, y_pred):
        eps = 1e-9
        N = len(y_true)
        da=(1/N) * (-(y_true / (y_pred + eps)) + (1 - y_true) / (1 - y_pred + eps))  # ∂L/∂ŷ o ∂L/∂a
        return da # (batch,)
       
class CrossEntropy:
   # ── Loss function: Cross-Entropy ──────
   def forward(self, y_true, y_pred): # y_true : (batch,n_classes) y_pred : (batch,n_classes)
       eps = 1e-9
       return np.mean(-np.sum(y_true * np.log(y_pred + eps),axis=1))  # average loss over the batch 

   def backward(self, y_true, y_pred):
        eps = 1e-9
        N = len(y_true)
        da=(1/N) * (y_pred -y_true)     # ∂L/∂ŷ o ∂L/∂a
        return da # (batch, n_classes)
    
class Sigmoid:
   def forward(self, z):   # z: (batch,1)
        self.a = 1 / (1 + np.exp(-z))
        return self.a
   def backward(self):
       return self.a*(1- self.a) # derivative of sigma = sigma*(1-sigma)  . Shape: #(batch,1)

class Softmax:
    def forward(self, z):   # z: (batch, n_classes)
         # operates on axis=1
        e=np.exp(z-np.max(z,axis=1).reshape(z.shape[0],1))
        self.a=np.divide(e,np.sum(e,axis=1).reshape(z.shape[0],1))
        return self.a
          
    def backward(self):
        # Gradient fused with CrossEntropy loss.
        # Only valid when paired with CrossEntropy — do not use with other loss functions.
        return 1
class Dense:

    def __init__(self, inputs, neurons, activation_function, lr=0.1, seed=None):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W =  rng.standard_normal((neurons, inputs)) * 0.01  # (n_neurons,n_inputs)
        self.b = np.zeros(neurons)          # (n_neurons,)
        self.activation_function = activation_function
        self.xb = None
        self.a = None

    def forward(self, layer_inputs):
        self.xb=layer_inputs # save inputs
        z=(layer_inputs @ self.W.T )+ self.b # (batch,n_neurons)
        self.a=self.activation_function.forward(z)  #(batch,n_neurons)
        return self.a

    def backward(self, da):
        dz = da * self.activation_function.backward()   #(batch, n_neurons)
        dW = dz.T @ self.xb   # weigth gradient  (neurons, inputs) — no dividtion by len(self.xb)  before:dW = dz.T @ self.xb / len(self.xb)   
        db = np.sum(dz, axis=0)   #bias gradient=   → shape (n_neurons,) sum instead mean  before: db = np.mean(dz, axis=0)
        da_prev = dz @ self.W             
        self.W -= self.lr * dW  
        self.b -= self.lr * db

        return da_prev
        
   
        
class MLP: # Multilayer Perceptron
    def __init__(self, layers, loss_function):
        self.layers=layers
        self.loss_function = loss_function
        self.losses = []
        
    def predict(self, X):
        return self.forward(X)
    
    def forward(self, X):
       layer_inputs=X
       for layer in self.layers:
          layer_outputs=layer.forward(layer_inputs)
          layer_inputs=layer_outputs
       return layer_outputs  #(batch,last_layer_n_neurons)

    def _train_one_epoch(self, X, y, batch_size=1):
        epoch_loss = 0
        for start in range(0, len(X), batch_size):
            xb = X[start:start+batch_size]  # x batch (batch, features)
            yb = y[start:start+batch_size]  # y batch (batch,)
            y_pred = self.forward(xb) # (batch,last_layer_n_neurons)
            loss = self.loss_function.forward(yb, y_pred)
            epoch_loss += loss
            # Backward
            da = self.loss_function.backward(yb, y_pred)
            for layer in reversed(self.layers):
                da = layer.backward(da)
        
        self.losses.append(epoch_loss)    
        return epoch_loss

    def train(self, X, y, epochs=100, batch_size=1):
        for epoch in range(epochs):
            epoch_loss = self._train_one_epoch(X, y, batch_size)
        return self.losses




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
        dw = xb.T @ dz          #   weigth gradient (feature,) before:  dw = xb.T @ dz / len(xb) 
        db = np.sum(dz, axis=0) # bias gradient= batch mean  → shape (n_neurons,)  before:  db = np.mean(dz, axis=0) 
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