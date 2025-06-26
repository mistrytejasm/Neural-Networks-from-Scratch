# --------- Input Data ---------

# Input values for a single sample (4 features)
inputs = [1, 2, 3, 2.5]

# --------- Weights and Biases ---------

# Weights for a layer of 3 neurons.
# Each inner list represents the weights for one neuron (corresponding to 4 inputs).
weights = [
    [0.2, 0.8, -0.5, 1.0],      # Weights for neuron 1
    [0.5, -0.91, 0.26, -0.5],   # Weights for neuron 2
    [-0.26, -0.27, 0.17, 0.87]  # Weights for neuron 3
]

# Biases for each of the 3 neurons
biases = [2, 3, 0.5]

# --------- Forward Pass (Manual Computation) ---------

# Empty list to store the output of each neuron
layer_outputs = []

# Loop through each neuron’s weights and its corresponding bias
for neuron_weights, neuron_bias in zip(weights, biases):
    # Start with a neuron output of 0
    neuron_output = 0

    # Compute the weighted sum of inputs for this neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight

    # Add the neuron's bias to the weighted sum
    neuron_output += neuron_bias

    # Store the final output of the neuron
    layer_outputs.append(neuron_output)

# --------- Output ---------

# Print the output values from all neurons in this layer
print(layer_outputs)
