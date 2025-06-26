import numpy as np

# --------- Input Data ---------

# Each inner list is a data sample with 4 features
inputs = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

# --------- First Layer (Layer 1) ---------

# Weights for the first layer: 3 neurons, each with 4 input weights
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

# Biases for the 3 neurons in the first layer
biases = [2, 3, 0.5]

# Perform dot product of inputs and transposed weights, then add biases
# Result: output from the first layer (shape: 3 samples × 3 neurons)
layer1_output = np.dot(inputs, np.array(weights).T) + biases

# --------- Second Layer (Layer 2) ---------

# Weights for the second layer: 3 neurons, each taking 3 inputs (from layer 1's output)
weights2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]
]

# Biases for the 3 neurons in the second layer
biases2 = [-1, 2, -0.5]

# The output of the first layer becomes the input to the second layer
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

# --------- Final Output ---------

# Print the output from the second layer (3 samples × 3 neurons)
print(layer2_output)
