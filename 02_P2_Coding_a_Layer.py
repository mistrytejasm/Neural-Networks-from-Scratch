# --------- Input Data ---------

# Input values for a single data sample (4 features)
inputs = [1, 2, 3, 2.5]

# --------- Weights for Each Neuron ---------

# Weights for neuron 1 — one weight for each input
weights1 = [0.2, 0.8, -0.5, 1.0]

# Weights for neuron 2
weights2 = [0.5, -0.91, 0.26, -0.5]

# Weights for neuron 3
weights3 = [-0.26, -0.27, 0.17, 0.87]

# --------- Biases for Each Neuron ---------

# Bias for neuron 1
bias1 = 2

# Bias for neuron 2
bias2 = 3

# Bias for neuron 3
bias3 = 0.5

# --------- Manual Forward Pass ---------

# Manually compute the output of each neuron:
# output = (input1 × weight1) + (input2 × weight2) + ... + bias

output = [
    # Output for neuron 1
    inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] +
    inputs[3] * weights1[3] + bias1,

    # Output for neuron 2
    inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] +
    inputs[3] * weights2[3] + bias2,

    # Output for neuron 3
    inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] +
    inputs[3] * weights3[3] + bias3
]

# --------- Final Output ---------

# Print the output of all 3 neurons
print(output)
