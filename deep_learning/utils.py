import numpy as np
import matplotlib.pyplot as plt
def train_test_split(X, y, test_size=0.2, seed=None):
    rng = np.random.default_rng(seed=seed)
    indexes = rng.permutation(len(X))
    n_test = int(len(X) * test_size)
    return (
        X[indexes[n_test:]], X[indexes[:n_test]],
        y[indexes[n_test:]], y[indexes[:n_test]],
        )
    
def plot_learning_curve(errors, ylabel):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(errors) + 1), errors, marker='o', color='b')
    plt.title('Error evolution (Learning Curve)')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def plot_neuron_heatmap(model):
    w = model.w.reshape(1, -1)
    vmax = np.abs(w).max()
    plt.figure(figsize=(10, 1))
    plt.imshow(w, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.yticks([])
    plt.title("Weights")
    plt.show()