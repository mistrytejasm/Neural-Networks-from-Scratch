import numpy as np

# Set a random seed for reproducibility
np.random.seed(0)

# Input data: each inner list is one sample with 4 features
X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

# Define a class for a fully connected (dense) layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with small random values
        # Shape: (number of inputs, number of neurons)
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)

        # Initialize biases with zeros
        # Shape: (1, number of neurons) — one bias per neuron
        self.biases = np.zeros((1, n_neurons))

    # Forward pass method to compute output of the layer
    def forward(self, inputs):
        # Calculate output values: dot product of inputs and weights, plus biases
        self.output = np.dot(inputs, self.weights) + self.biases


# Create the first dense (hidden) layer:
# - Input has 4 features
# - This layer has 5 neurons → outputs 5 values per sample
layer1 = Layer_Dense(4, 5)

# Create the second dense layer:
# - Takes input from layer1's 5 outputs
# - This layer has 2 neurons → outputs 2 values per sample
layer2 = Layer_Dense(5, 2)

# Perform the forward pass through the first layer
layer1.forward(X)

# Perform the forward pass through the second layer
# using the output from the first layer as input
layer2.forward(layer1.output)

# Print the final output from the second layer
print(layer2.output)
