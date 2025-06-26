"""
This script demonstrates two concepts:

1. **Solving for exponentials using natural log:**
   If e^x = b, then x = ln(b). This shows how we can solve for x.

2. **Calculating categorical cross-entropy loss:**
   This is commonly used in classification tasks to compare the predicted
   probabilities (from softmax) against the true target class.
"""

import math

# --------------------------
# Solving for x in e^x = b
# --------------------------
# Uncomment below lines to understand how logarithms work with exponentials

# import numpy as np
# b = 5.2
# print(np.log(b))                         # Natural log of 5.2
# print(math.e ** 1.6486586255873816)     # Should return ~5.2, proving e^ln(b) = b

# -----------------------------------------------
# Categorical Cross-Entropy Loss Calculation
# -----------------------------------------------

# Predicted probabilities from a softmax function
softmax_output = [0.7, 0.1, 0.2]  # Model is 70% confident in class 0

# Actual (target) class is class 0 → One-hot encoded as [1, 0, 0]
target_output = [1, 0, 0]

# Manually computing categorical cross-entropy loss
# Only the log of the probability of the true class (class 0) is used
loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print("Loss (manual calculation):", loss)

# This is the simplified form, since only the true class (index 0) contributes
loss = -math.log(softmax_output[0])
print("Loss (simplified):", loss)

# Additional examples for intuition:
# Higher confidence in the correct class → lower loss
print("Loss when prediction = 0.7:", -math.log(0.7))  # lower loss
print("Loss when prediction = 0.5:", -math.log(0.5))  # higher loss
