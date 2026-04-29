
import numpy as np

#-----------ACTIVATION FUNCTIONS-------------------------
def step(z):
    # Heaviside step function: returns 1 if z >= 0, else 0.
    return (z >= 0).astype(int)

class ReLU:
   def forward(self, z):
        self.z = z
        return np.maximum(0,z)
   def backward(self):
       return (self.z > 0).astype(int) #  da · ∂a/∂z   Shape: #(batch,1)
   
   def init_std(self, fan_in): #fan_in is input size
       return np.sqrt(2 / fan_in)   # He initializacion
class Sigmoid:
   def forward(self, z):   # z: (batch,1)
        self.a = 1 / (1 + np.exp(-z))
        return self.a
   def backward(self):
       return self.a*(1- self.a) # derivative of sigma = sigma*(1-sigma)  . Shape: #(batch,1)
   def init_std(self, fan_in): #fan_in is input size
        return np.sqrt(1 / fan_in)   # Xavier initializacion
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
        
    def init_std(self, fan_in): #fan_in is input size
            return np.sqrt(1 / fan_in)   # Xavier initialization
        
#-----------LOSS FUNCTIONS-------------------------        
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
#------------------------------------------------------    

#----------LAYERS---------
class Flatten:
    def __init__(self):
        print(f'Flatten layer')
        
    def forward(self, layer_inputs):  #(n, 16, 5, 5) (N,C,H,W)
        self.original_shape=layer_inputs.shape #save original shape for backward
        #print(f'flatten input shape  {layer_inputs.shape}')
        ret = layer_inputs.reshape(layer_inputs.shape[0],-1)
        #print(f'flatten output shape  {ret.shape}')
        return ret
        
    def backward(self, da): #da shape (n, 400)
        da=da.reshape(self.original_shape)
        return da
        
class AVGPooling:

    def __init__(self,kernel_size):
        self.kernel_size=kernel_size
        print(f'AVGPooling layer with kernel size: {self.kernel_size[0]} x {self.kernel_size[1]}')

    def forward(self, layer_inputs):     #layer_inputs shape=(n, 6, 28, 28))  
        n_batch=layer_inputs.shape[0]
        self.pool = np.zeros((n_batch, layer_inputs.shape[1],int(layer_inputs.shape[2]/ self.kernel_size[0]),int(layer_inputs.shape[3]/ self.kernel_size[1])))
        row=0
        col=0

        for c in range(self.pool.shape[1]):
            for row in range(self.pool.shape[2]): 
                for col in range(self.pool.shape[3]): 
                    input_slice=layer_inputs[:,c,row*self.kernel_size[0]:row*self.kernel_size[0]+self.kernel_size[0], col*self.kernel_size[1]: col*self.kernel_size[1]+self.kernel_size[1]]
                    self.pool[:,c,row,col]=np.mean(input_slice)
        return self.pool

    def backward(self, da):  
        new_da= np.zeros((da.shape[0], da.shape[1],da.shape[2]* self.kernel_size[0],da.shape[3]* self.kernel_size[1]))
        #print(f'AVGPooling backward receives shape {da.shape}')#(100, 16, 5, 5)
        #print(f'AVGPooling backward receives  {da[0,0]}')#(100, 16, 5, 5)
        for n in range(new_da.shape[0]):
            for c in range(new_da.shape[1]):
                for row in range(new_da.shape[2]): 
                    for col in range(new_da.shape[3]): 
                        #print(f'AVGPooling accediendo a posicion  {row} {col} {int(row/da.shape[2])} {int(col/da.shape[3])}')
                        new_da[n,c,row,col]=da[n,c,int(row/self.kernel_size[0]),int(col/self.kernel_size[1])]/(self.kernel_size[0]*self.kernel_size[1])
        #print(f'AVGPooling backward devuelve shape {new_da.shape}')  
        #print(f'AVGPooling backward devuelve  {new_da[0,0]}')#(100, 16, 5, 5)
        #print(f'AVGPooling se dividió por  {self.kernel_size[0]*self.kernel_size[1]}')
        return new_da
      
class Convolution2D: # filter/kernel slides in 2 dimensions (Height and Width).

    def __init__(self,  kernel_size, n_kernels, stride, input_shape, padding, activation_function, lr=0.1, seed=None):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.input_shape = input_shape
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size 
        #self.kernels = rng.standard_normal((n_kernels,input_shape[0], kernel_size[0], kernel_size[1])) * 0.01
        self.kernels = rng.standard_normal((n_kernels,input_shape[0], kernel_size[0], kernel_size[1])) * activation_function.init_std(input_shape[0] * kernel_size[0] * kernel_size[1])

        self.stride = stride
        self.padding = padding
        self.activation_function = activation_function
        self.b = np.zeros(n_kernels)          # (n_kernels,) one bias per filter/kernel
        self.act_map_rows=int(((input_shape[1]-self.kernel_size[0]+2*self.padding)/self.stride)+1)
        self.act_map_cols=int(((input_shape[2]-self.kernel_size[1]+2*self.padding)/self.stride)+1)
        parameters=input_shape[0] * n_kernels* kernel_size[0]* kernel_size[1]+len(self.b);
        neurons=n_kernels* self.act_map_rows* self.act_map_cols 
        np.set_printoptions(precision=3, suppress=True,linewidth=100)
        print(f'Convolutional 2D layer with: {parameters} trainable parameters, {neurons} neurons, {parameters*self.act_map_rows* self.act_map_cols} connections, Input: {input_shape},  Output: {(self.n_kernels,self.act_map_rows, self.act_map_cols)}')
        
               
    def forward(self, layer_inputs):#(N,1,28,28) (N,6, 28, 28) NCHW  (N,C,H,W) N=batch_size C=channel, H=height W=width
     
        n_batch=layer_inputs.shape[0]
        self.feature_map = np.zeros((n_batch, self.n_kernels, self.act_map_rows,self.act_map_cols)) # feature map zero-initialized
        self.padded_inputs = np.pad(layer_inputs, pad_width=((0, 0),(0, 0), (self.padding, self.padding),(self.padding,self.padding)), mode='constant', constant_values=0) #padding only H and W .Shape (2, 1, 32, 32))
        self.xb_padded=self.padded_inputs # save padded inputs
        #print(f'Convolutional 2D layer forward with input:\n { self.xb_padded} \n and kernels:\n {self.kernels}');
        row=0
        col=0
    
       
        for row in range(0,self.act_map_rows):
            for col in range(0,self.act_map_cols):
                row_input=row*self.stride
                col_input=col*self.stride
                input_slice=self.padded_inputs[:,np.newaxis,:,row_input:row_input+self.kernel_size[0], col_input: col_input+self.kernel_size[1]] #add k dimension
                #print(f'slice shape {input_slice.shape}')
                #print(f'kernels shape {self.kernels[np.newaxis,:].shape}')
                self.feature_map[:,:,row,col]=(input_slice * self.kernels[np.newaxis,:,:,:]).sum(axis=(2,3,4))+self.b[np.newaxis,:] #add batch dimension in kernels and then assigning something with shape (N,n_kernels)
        
        #print(f'Convolutional 2D layer forward , feature maps:\n { self.feature_map}');  
        self.activation_map=self.activation_function.forward(self.feature_map)
       
        return self.activation_map

    def backward(self, da):
        n_batch=da.shape[0]
        #print(f'Convolution  backward receives shape {da.shape}')#(100, 16, 10, 10)
        dz = da * self.activation_function.backward()   #Feature map gradients (batch, n_kernels , h_out, w_out) 
        #print(f'Convolution  dz  shape {dz.shape}')#(100, 16, 10, 10) (batch, n_kernels , h_out, w_out) 
        
       
      
        da_prev_padded= np.zeros(self.xb_padded.shape) #(n_batch, C, H, W)
        #print(f'shape da_prev = {da_prev.shape}') 
        #kernels_copy = self.kernels.copy()#rotated_kernels=np.flip(self.kernels, axis=(2, 3))
        for c in range(self.kernels.shape[1]):
            for row_kernel in range(self.kernel_size[0]):
                for col_kernel in range(self.kernel_size[1]):
                      for row in range(0,self.act_map_rows,self.stride):
                            for col in range(0,self.act_map_cols,self.stride):
                                 row_input = row  + row_kernel
                                 col_input = col + col_kernel
                                 #print(f'Ubico en da_prev {row*self.stride+row_kernel} {col*self.stride+col_kernel}')
                                 da_prev_padded[:,c,row_input,col_input]+= dz[:,:,row,col] @self.kernels[:,c,row_kernel,col_kernel] #sliding from back to top
                                                                          
        if self.padding > 0:
            #print(f'Hay padding, paso de  {da_prev_padded.shape}  a {da_prev_padded[:, :, self.padding:-self.padding, self.padding:-self.padding].shape}')
            da_prev = da_prev_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            da_prev = da_prev_padded
         
        #print(f'Calculating Kernel gradient ...')
        dK =   np.zeros(self.kernels.shape) #inicialized in zero for gradient accumulation shape (n_kernels, c, self.kernel_size[0], self.kernel_size[1])
        #print(f'Convolution  dK  shape {dK.shape}')
        #kernel gradiente calculation
        for c in range(dK.shape[1]):
                    for row_kernel in range(self.kernel_size[0]):
                        for col_kernel in range(self.kernel_size[1]):
                              for row in range(0,self.act_map_rows,self.stride):
                                    for col in range(0,self.act_map_cols,self.stride):
                                        #print(f'Convolution  Acumulo en  {row_kernel} {col_kernel} lo que esta en {row+row_kernel} {col+col_kernel}')
                                        dK[:,c,row_kernel,col_kernel]+= dz[:,0:self.n_kernels,row,col].T@ self.xb_padded[:,c,row*self.stride+row_kernel,col*self.stride+col_kernel]   #∂L/∂K[k,c,m,n] = Σₙ Σᵢ Σⱼ X_padded[n,c,i+m,j+n] · dz[n,k,i,j]
                                        
        #print(f' dk = {dK} ') 
        #print(f'Updating weights ...')
        db = np.sum(dz, axis=(0,2,3))   # bias gradient=   → shape (n_kernels,) sum instead mean  before: db = np.mean(dz, axis=(0,2,3))
        self.kernels -= self.lr * dK  
        self.b -= self.lr * db
        
        
               
        
        return da_prev

class Dense:

    def __init__(self, inputs, neurons, activation_function, lr=0.1, seed=None):
        rng = np.random.default_rng(seed)
        self.lr = lr
        #self.W =  rng.standard_normal((neurons, inputs)) * 0.01  # (n_neurons,n_inputs)
        self.W = rng.standard_normal((neurons, inputs)) * activation_function.init_std(inputs)
        self.b = np.zeros(neurons)          # (n_neurons,)
        self.activation_function = activation_function
        self.xb_padded = None
        self.a = None
        print(f'Dense layer with: {neurons*inputs+neurons} trainable parameters, {neurons} neurons, {neurons*inputs+neurons} connections')

    def forward(self, layer_inputs):
        self.xb=layer_inputs # save inputs
        z=(layer_inputs @ self.W.T )+ self.b # (batch,n_neurons)
        self.a=self.activation_function.forward(z)  #(batch,n_neurons)
        return self.a

    def backward(self, da):
        dz = da * self.activation_function.backward()   #(batch, n_neurons)
        #print(f'{dz.T.shape} { self.xb.shape}')
        dW = dz.T @ self.xb   # weigth gradient  (neurons, inputs) — no dividtion by len(self.xb)  before:dW = dz.T @ self.xb / len(self.xb)   
        db = np.sum(dz, axis=0)   #bias gradient=   → shape (n_neurons,) sum instead mean  before: db = np.mean(dz, axis=0)

        #da for previous layer
        da_prev = dz @ self.W   
        
        #weights update
        self.W -= self.lr * dW  
        self.b -= self.lr * db

        
          
        return da_prev
        
   
        
class SequentialNetwork: 
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

    def _train_one_epoch(self, X, y, batch_size=1,shuffle=False):
        epoch_loss = 0
        if shuffle:
            # Shuffle al inicio de cada epoch
            idx = np.random.permutation(len(X))
            X, y = X[idx], y[idx]
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

    def train(self, X, y, epochs=100, batch_size=1,shuffle=False):
        for epoch in range(epochs):
            epoch_loss = self._train_one_epoch(X, y, batch_size,shuffle)
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