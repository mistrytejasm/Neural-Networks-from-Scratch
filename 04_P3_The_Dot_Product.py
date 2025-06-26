import numpy as np

# --------- Inputs to the Neuron ---------

# Input values for a single sample (4 features)
inputs = [1, 2, 3, 2.5]

# --------- Weights and Bias ---------

# Weights for a single neuron — one weight per input
weights = [0.2, 0.8, -0.5, 1.0]

# Bias term for the neuron
bias = 2

# --------- Forward Pass Calculation ---------

# Compute the output of the neuron:
# - Use dot product of inputs and weights
# - Add the bias term
# np.dot(inputs, weights) calculates:
#   (1×0.2) + (2×0.8) + (3×-0.5) + (2.5×1.0)
output = np.dot(inputs, weights) + bias

# --------- Final Output ---------

# Print the neuron's output
print(output)
