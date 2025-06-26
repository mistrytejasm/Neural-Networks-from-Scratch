# This script demonstrates a simplified and efficient implementation of the softmax
# activation function using NumPy. Softmax is used in neural networks, especially in
# classification problems, to convert raw model outputs (logits) into probabilities
# that sum up to 1.

import math       # To access Euler's number (e), though NumPy also handles this internally
import numpy as np  # NumPy is used here for efficient numerical computations

# Simulated raw outputs (logits) from a neural network layer
layer_outputs = [4.8, 1.21, 2.385]

# You can manually use Euler's number like this:
# E = 2.71828182846
# But it's better to use NumPy's exp function which uses math.e internally

# Step 1: Apply the exponential function to each output
# NumPy does this efficiently on the entire list (vectorized operation)
exp_values = np.exp(layer_outputs)  # Equivalent to [e**x for x in layer_outputs]

# Step 2: Normalize the exponential values by dividing each one by the sum of all exponentials
# This converts the values into a probability distribution
norm_values = exp_values / np.sum(exp_values)

# Print the normalized values, which represent the output probabilities
print("Normalize: ", norm_values)

# Print the sum of the normalized values; it should be 1.0 (or very close due to floating point precision)
print("Normalize Sum: ", sum(norm_values))

# ✅ Summary:
# The combination of exponentiation and normalization is what forms the **softmax activation function**.
# It's commonly used in the output layer of classification models to represent class probabilities.
