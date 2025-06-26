# This script demonstrates how to manually implement the softmax function,
# a key concept in neural networks used to convert raw output values (logits)
# into probabilities that sum to 1.

import math  # Importing math module to access the value of Euler's number (e)

# Simulated raw outputs from a neural network layer (logits)
layer_outputs = [4.8, 1.21, 2.385]

# Get the value of Euler's number (approx. 2.71828) from the math module
E = math.e

# Step 1: Apply the exponential function to each value in the layer output
# This amplifies larger values and helps make them more separable
exp_values = []  # List to store exponential values
for output in layer_outputs:
    exp_values.append(E ** output)  # Equivalent to math.exp(output)

# Print the exponential values for understanding
print("Exponential Value: ", exp_values)

# Step 2: Normalize the exponential values to get probabilities
# This is done by dividing each exponential by the sum of all exponentials
norm_base = sum(exp_values)  # Sum of all exponential values
norm_values = []  # List to store normalized (probability) values
for value in exp_values:
    norm_values.append(value / norm_base)  # Normalization step

# Print the normalized values which now represent probabilities
print("Normalize value: ", norm_values)

# Print the sum of normalized values (should be 1.0 or very close due to floating point precision)
print("Normalize Sum: ", sum(norm_values))
