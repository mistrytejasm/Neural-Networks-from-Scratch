import numpy as np

# --------- Input Data ---------

# Single input sample with 4 features
inputs = [1, 2, 3, 2.5]

# --------- Layer Weights and Biases ---------

# Weights for a layer with 3 neurons, each taking 4 inputs
weights = [
    [0.2,  0.8, -0.5,  1.0],    # Weights for neuron 1
    [0.5, -0.91, 0.26, -0.5],   # Weights for neuron 2
    [-0.26, -0.27, 0.17, 0.87]  # Weights for neuron 3
]

# Biases for the 3 neurons in the layer
biases = [2, 3, 0.5]

# --------- Forward Pass ---------

# Compute the output of the layer:
# - For each neuron, perform a dot product of its weights with the inputs
# - Then add the corresponding bias
# - This is vectorized using NumPy: (3x4 weights) ⋅ (4x1 inputs) + (3x1 biases)
output = np.dot(weights, inputs) + biases

# --------- Output ---------

# Print the result — this is the output from the 3 neurons for the given input
print(output)
