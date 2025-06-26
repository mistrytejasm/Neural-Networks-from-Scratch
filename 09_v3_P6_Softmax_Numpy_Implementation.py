# This script demonstrates how to apply the softmax activation function to a **batch of outputs**
# from a neural network layer using NumPy. It also addresses a critical issue—**numerical overflow**
# when computing exponentials of large numbers—and shows how to prevent it.

import math
import numpy as np

# Simulated batch of raw outputs (logits) from a neural network
# Each sublist represents one sample (e.g., one input image or data point)
layer_outputs = [
    [4.8, 1.21, 2.385],
    [8.9, -1.81, 0.2],
    [1.41, 1.051, 0.026]
]

# Step 1: Exponentiation of the outputs
# This amplifies differences between values, making the biggest ones dominate
exp_values = np.exp(layer_outputs)

# Step 2: Normalize each sample's outputs so they sum to 1
# axis=1 means we are summing across columns (i.e., per row/sample)
# # axis = 0 is sum of columns and axis = 1 is sum of row
# keepdims=True keeps the 2D shape so that broadcasting works during division
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Print the normalized values (i.e., the softmax probabilities)
print("Normalize Values: ")
print(norm_values)

# ---------------------------------------------------------------
# ⚠️ Numerical Stability Issue: Overflow in Exponentials ⚠️
# ---------------------------------------------------------------
# The exponential function grows very rapidly. Large inputs like 100, 1000, or more
# can result in huge numbers (or even `inf`), causing overflow errors.

# ✅ Overflow Prevention Technique:
# Subtract the maximum value in each row from all values in that row *before* exponentiating.
# This doesn't affect the final softmax result because it's scale-invariant.
# Example:
#   Original: [5.0, 2.0, 1.0]
#   Max = 5.0 → Subtract 5.0 from each → [0.0, -3.0, -4.0]
#   This keeps numbers small, avoids overflow, and improves numerical stability.



# not show in video
# Here's how you can implement this properly:

# Step 1: Subtract max value from each row
adjusted_outputs = layer_outputs - np.max(layer_outputs, axis=1, keepdims=True)

# Step 2: Apply exponential function
stable_exp_values = np.exp(adjusted_outputs)

# Step 3: Normalize as usual
stable_norm_values = stable_exp_values / np.sum(stable_exp_values, axis=1, keepdims=True)

print("\nStable Normalize Values (with overflow prevention): ")
print(stable_norm_values)

# ✅ These values are now more stable and accurate for larger input values.
