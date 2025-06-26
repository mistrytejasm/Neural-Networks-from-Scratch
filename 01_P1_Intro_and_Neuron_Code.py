# --------- Input Data ---------

# A single input sample with 3 features
inputs = [1.2, 5.1, 2.1]

# --------- Weights and Bias ---------

# Weights for a single neuron (one weight per input)
weights = [3.1, 2.1, 8.7]

# Bias term for the neuron
bias = 3

# --------- Forward Pass for a Single Neuron ---------

# Calculate the output of the neuron:
# Multiply each input by its corresponding weight and add them all up,
# then add the bias
output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias

# --------- Final Output ---------

# Print the final result (output of the neuron)
print(output)
