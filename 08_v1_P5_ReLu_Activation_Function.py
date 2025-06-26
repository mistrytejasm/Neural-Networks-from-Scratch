import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Initialize NNFS (sets random seed and default data type for reproducibility)
nnfs.init()

# --------- Data Generation ---------

# Example manual input (not used here)
# X = [
#     [1, 2, 3, 2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5, 2.7, 3.3, -0.8]
# ]

# Generate a dataset of 100 samples per class, 3 classes — shaped like spirals
# X contains input features (shape: 300x2), y contains class labels (not used yet)
X, y = spiral_data(100, 3)

# --------- Dense (Fully Connected) Layer Class ---------

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with small random values
        # Shape: (number of inputs, number of neurons)
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)

        # Initialize biases to zeros
        # Shape: (1, number of neurons) — one bias per neuron
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Compute output of the layer:
        # output = inputs × weights + biases
        self.output = np.dot(inputs, self.weights) + self.biases

# --------- ReLU Activation Function Class ---------

class Activation_ReLu:
    def forward(self, inputs):
        # Apply ReLU activation: replace all negative values with 0
        self.output = np.maximum(0, inputs)

# --------- Neural Network Forward Pass ---------

# Create the first dense layer:
# - Takes 2 inputs (from the spiral data)
# - Has 5 neurons (produces 5 outputs per sample)
layer1 = Layer_Dense(2, 5)

# Create a ReLU activation function to introduce non-linearity
activation1 = Activation_ReLu()

# Perform the forward pass through the dense layer
layer1.forward(X)

# Pass the output of the dense layer through the ReLU activation
activation1.forward(layer1.output)

# Print the output after ReLU activation
print(activation1.output)
