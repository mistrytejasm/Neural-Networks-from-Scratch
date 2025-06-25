
# These are the input values to the neural network layer 
inputs = [1, 2, 3, 2.5]

# Each sublist is the weight vector for a neuron. There are 3 neurons, each with 4 weights (since we have 4 inputs).
weights = [[0.2,0.8,-0.5, 1.0], 
           [0.5,-0.91, 0.26,-0.5], 
           [-0.26, -0.27, 0.17, 0.87]]

# These are the bias terms added to each neuron's output.
biases = [2,3,0.5]


layer_outputs = []

# Iterates through each neuron's weights and its corresponding bias.
for neuron_weights, neuron_bias in zip(weights, biases):
  neuron_output = 0

  for n_input, weight in zip(inputs, neuron_weights):
    neuron_output += n_input*weight  # Computes the weighted sum of inputs for one neuron.
  neuron_output += neuron_bias       # Adds the bias and stores the final output of that neuron.
  layer_outputs.append(neuron_output)

print(layer_outputs)
