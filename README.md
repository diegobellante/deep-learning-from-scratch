# Deep Learning from Scratch

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
│   └── model.py          # Core implementations
│   └── utils.py          # Data splitting and visualization utilities
├── notebooks/
│   ├── 01_a_perceptron.ipynb
│   ├── 01_b_perceptron_iris.ipynb
│   ├── 02_a_logistic_neuron_iris.ipynb
│   └── 03_mlp.ipynb (soon)
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
 

---

## Stack

| Tool | Role |
|------|------|
| Python 3.x | Language |
| NumPy | All numerical computation |
| Matplotlib | Visualization |
| Jupyter | Interactive notebooks |

---

## Author

Built as a structured self-study project to develop deep, mechanical understanding of modern deep learning — from the first linear threshold unit to large language models.

## License

MIT © Diego Bellante
