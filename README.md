# Deep Learning from Scratch [![GitHub repo](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/diegobellante/deep-learning-from-scratch)

> Building every component by hand — no frameworks, no shortcuts.

This repository documents a progressive, ground-up implementation of deep learning architectures using only **Python and NumPy**. Each model is built from first principles: forward pass, backpropagation, and weight updates derived and coded manually.

The goal is not to train state-of-the-art models. The goal is to understand exactly what happens inside them.

---

## Planned Architectures

- Rosenblatt Perceptron
- Logistic Neuron
- Multilayer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network / LSTM
- Transformer
- Diffusion Model
- Large Language Model (LLM)

---

## Project Structure

```
deep-learning-from-scratch/
├── deep_learning/
│   ├── model.py          # Core implementations
│   └── utils.py          # Data splitting and visualization utilities
├── notebooks/
│   ├── 01_a_perceptron.ipynb
│   ├── 01_b_perceptron_iris.ipynb
│   ├── 02_a_logistic_neuron_iris.ipynb
│   ├── 03_a_MLP_iris.ipynb
│   ├── 03_b_MLP_mnist.ipynb
│   ├── 04_a_cnn_shallow_uci_digits.ipynb
│   └── 04_b_cnn_lenet_5_mnist.ipynb
└── README.md
```

---

## Requirements & Installation

**Python 3.x** and NumPy are the only hard dependencies.

```bash
git clone https://github.com/diegobellante/deep-learning-from-scratch.git
cd deep-learning-from-scratch
pip install numpy matplotlib
```

To run the notebooks:

```bash
pip install jupyter
jupyter lab notebooks/
```


## What's Implemented

### 01 · Rosenblatt Perceptron

The original 1958 binary classifier. Implements Rosenblatt's correction rule — weight updates happen per sample, not per batch.

- Heaviside step activation
- Online learning (one update per sample)
- Convergence guaranteed for linearly separable data
- Animated decision boundary visualization

```python
from deep_learning.model import Perceptron

model = Perceptron(n_features=2, lr=0.1, seed=42)
errors = model.train(X, y, epochs=100)
predictions = model.predict(X)
```
 
### 02 · Logistic Neuron
 
A single neuron with sigmoid activation and binary cross-entropy loss, trained with gradient descent. Architecturally identical to a one-layer MLP — the conceptual bridge between the Perceptron and deep networks.
 
- Sigmoid activation + BCE loss
- Mini-batch gradient descent with configurable batch size
- Parametric design: swap activation and loss functions without changing the neuron
- Validated on Iris (setosa vs versicolor) — same dataset as the Perceptron, direct comparison of learning rules
 
```python
from deep_learning.model import Neuron, Sigmoid, BCE
 
model = Neuron(n_features=4, activation_function=Sigmoid(), loss_function=BCE(), lr=0.1, seed=42)
losses = model.train(X_train, y_train, epochs=100)
probabilities = model.predict(X_test)
```
 
### 03 · Multilayer Perceptron (MLP)
 
A feedforward fully connected network with arbitrary depth, trained with backpropagation derived from the chain rule. Demonstrates the limitations of single-neuron models — Iris (3 classes) is not linearly separable, and the Logistic Neuron cannot solve it.
 
- Modular layer design: stack any number of `Dense` layers with any activation
- ReLU in hidden layers, Softmax in the output layer
- Softmax + CrossEntropy loss (fused gradient for numerical stability)
- Validated on Iris (3 classes) — direct comparison with the Logistic Neuron
 
```python
from deep_learning.model import Dense, MLP, ReLU, Softmax, CrossEntropy
 
model = MLP(layers=[
    Dense(inputs=X_train.shape[1], neurons=8, activation_function=ReLU(), lr=LR, seed=SEED),
    Dense(inputs=8, neurons=3, activation_function=Softmax(), lr=LR, seed=SEED)
], loss_function=CrossEntropy())
losses = model.train(X_train, y_train_one_hot, epochs=2000, batch_size=8)
predictions = model.predict(X_test)
```

### 04 · Convolutional Neural Network (CNN)

A convolutional network built from scratch, implementing forward and backward passes through Conv2D, pooling, and dense layers. Validated on two benchmarks of increasing complexity.

- `Convolution2D` with configurable kernel size, stride, padding and number of kernels
- `AVGPooling` for spatial downsampling
- `Flatten` to bridge convolutional and dense layers
- Softmax + CrossEntropy output (same as MLP)
- Optimized convolution using vectorized operations (`fast=True`)

**Shallow CNN — UCI Digits (8×8 grayscale, 10 classes):**

```python
from deep_learning.model import SequentialNetwork, Convolution2D, Flatten, Dense, ReLU, Softmax, CrossEntropy

model = SequentialNetwork(layers=[
    Convolution2D(kernel_size=(3,3), n_kernels=16, input_shape=(1,8,8), stride=1, padding=1, activation_function=ReLU(), lr=LR, seed=SEED, fast=True),
    Flatten(),
    Dense(inputs=1024, neurons=64, activation_function=ReLU(), lr=LR, seed=SEED),
    Dense(inputs=64, neurons=10, activation_function=Softmax(), lr=LR, seed=SEED)
], loss_function=CrossEntropy())
```

**LeNet-5 — MNIST (28×28 grayscale, 10 classes):**

```python
model = SequentialNetwork(layers=[
    Convolution2D(kernel_size=(5,5), n_kernels=6, input_shape=(1,28,28), stride=1, padding=2, activation_function=ReLU(), lr=LR, seed=SEED, fast=True),
    AVGPooling(kernel_size=(2,2)),
    Convolution2D(kernel_size=(5,5), n_kernels=16, input_shape=(6,14,14), stride=1, padding=0, activation_function=ReLU(), lr=LR, seed=SEED, fast=True),
    AVGPooling(kernel_size=(2,2)),
    Flatten(),
    Dense(inputs=400, neurons=120, activation_function=ReLU(), lr=LR, seed=SEED),
    Dense(inputs=120, neurons=84, activation_function=ReLU(), lr=LR, seed=SEED),
    Dense(inputs=84, neurons=10, activation_function=Softmax(), lr=LR, seed=SEED)
], loss_function=CrossEntropy())
```



---

## Stack

| Tool | Role |
|------|------|
| Python 3.x | Language |
| NumPy | All numerical computation |
| Matplotlib | Visualization |
| Jupyter | Interactive notebooks |

---
[![GitHub repo](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/diegobellante/deep-learning-from-scratch)

## Author

Built as a structured self-study project to develop deep, mechanical understanding of modern deep learning — from the first linear threshold unit to large language models.

## License

MIT © Diego Bellante
