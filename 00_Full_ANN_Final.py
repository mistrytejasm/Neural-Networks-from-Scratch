"""
Complete Neural Network Implementation from Scratch

This script demonstrates a full neural network implementation including:
- Forward propagation through dense layers
- ReLU and Softmax activation functions
- Categorical cross-entropy loss calculation
- Backward propagation (backprop) for gradient computation
- Parameter updates using gradient descent
- Training loop with accuracy calculation
- Visualization of training progress

Dataset: Spiral dataset with 3 classes
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

def spiral_data(samples, classes):
    """Generate spiral dataset for classification"""
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

# ------------------------------------
# Dense (Fully Connected) Layer Class
# ------------------------------------
class Layer_Dense:
    """
    Dense layer implementation with forward and backward pass

    Parameters:
    - n_inputs: Number of input features
    - n_neurons: Number of neurons in this layer
    """
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with small random values (Xavier/Glorot initialization)
        # Shape: (n_inputs, n_neurons) - each column represents weights for one neuron
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        # Initialize biases to zero - one bias per neuron
        # Shape: (1, n_neurons) - broadcast across all samples
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """
        Forward pass: compute layer output

        Formula: output = inputs @ weights + biases
        - inputs: (batch_size, n_inputs)
        - weights: (n_inputs, n_neurons)
        - biases: (1, n_neurons)
        - output: (batch_size, n_neurons)
        """
        # Store inputs for backward pass
        self.inputs = inputs
        # Compute weighted sum + bias for each neuron
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        """
        Backward pass: compute gradients w.r.t. weights, biases, and inputs

        dvalues: gradient flowing from the next layer (batch_size, n_neurons)
        """
        # Gradients on weights: inputs^T @ dvalues
        # Shape: (n_inputs, n_neurons)
        self.dweights = np.dot(self.inputs.T, dvalues)

        # Gradients on biases: sum across batch dimension
        # Shape: (1, n_neurons)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on inputs: dvalues @ weights^T (for next layer's backward pass)
        # Shape: (batch_size, n_inputs)
        self.dinputs = np.dot(dvalues, self.weights.T)

# -----------------------
# ReLU Activation Function
# -----------------------
class Activation_ReLU:
    """
    ReLU (Rectified Linear Unit) activation function
    Forward: f(x) = max(0, x)
    Backward: f'(x) = 1 if x > 0, else 0
    """
    def forward(self, inputs):
        """Apply ReLU activation element-wise"""
        # Store inputs for backward pass
        self.inputs = inputs
        # ReLU: replace negative values with 0
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        """
        Backward pass for ReLU
        Gradient is 1 where input > 0, otherwise 0
        """
        # Copy dvalues to avoid modifying original
        self.dinputs = dvalues.copy()
        # Zero out gradients where input was <= 0
        self.dinputs[self.inputs <= 0] = 0

# --------------------------
# Softmax Activation Function
# --------------------------
class Activation_Softmax:
    """
    Softmax activation function for multi-class classification
    Converts raw scores to probability distribution
    """
    def forward(self, inputs):
        """
        Apply softmax activation
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
        """
        # Store inputs for backward pass
        self.inputs = inputs

        # Subtract max for numerical stability (prevents overflow)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize to get probabilities (sum to 1 across classes)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        """
        Backward pass for softmax
        Complex derivative due to normalization affecting all outputs
        """
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Calculate gradient for each sample individually
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten to column vector
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of softmax
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate gradient and store
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# ------------------------
# Base Loss Class
# ------------------------
class Loss:
    """Base class for loss functions"""
    def calculate(self, output, y):
        """Calculate mean loss across batch"""
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# ----------------------------------------
# Categorical Cross-Entropy Loss Function
# ----------------------------------------
class Loss_CategoricalCrossentropy(Loss):
    """
    Categorical Cross-Entropy Loss for multi-class classification
    Formula: L = -sum(y_true * log(y_pred))
    """
    def forward(self, y_pred, y_true):
        """
        Forward pass: calculate loss for each sample

        y_pred: predicted probabilities (batch_size, n_classes)
        y_true: true labels (sparse: batch_size,) or one-hot: (batch_size, n_classes)
        """
        samples = len(y_pred)

        # Clip predictions to prevent log(0) which would be -inf
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Handle sparse labels (integers like 0, 1, 2)
        if len(y_true.shape) == 1:
            # Extract predicted probability for correct class
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # Handle one-hot encoded labels ([1,0,0], [0,1,0], etc.)
        elif len(y_true.shape) == 2:
            # Sum across classes (only correct class contributes)
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Calculate negative log-likelihood
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        """
        Backward pass: calculate gradients w.r.t. predictions
        """
        samples = len(dvalues)
        labels = len(dvalues[0])  # number of classes

        # Convert sparse labels to one-hot if needed
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient: -y_true / y_pred
        self.dinputs = -y_true / dvalues
        # Normalize gradient by batch size
        self.dinputs = self.dinputs / samples

# ---------------------------------------------
# Combined Softmax + Cross-Entropy (Optimized)
# ---------------------------------------------
class Activation_Softmax_Loss_CategoricalCrossentropy():
    """
    Combined Softmax activation and Categorical Cross-entropy loss
    More numerically stable and computationally efficient than separate implementation
    """
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        """Forward pass through softmax and loss calculation"""
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        """
        Optimized backward pass
        Derivative of softmax + cross-entropy simplifies to: y_pred - y_true
        """
        samples = len(dvalues)

        # Convert sparse labels to one-hot if needed
        if len(y_true.shape) == 1:
            y_true = np.eye(len(dvalues[0]))[y_true]

        # Gradient is simply prediction - true label
        self.dinputs = dvalues - y_true
        # Normalize by batch size
        self.dinputs = self.dinputs / samples

# ------------------
# Optimizer Class
# ------------------
class Optimizer_SGD:
    """
    Stochastic Gradient Descent optimizer
    Updates parameters using: param = param - learning_rate * gradient
    """
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        """Update layer parameters using computed gradients"""
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

# ------------------
# Utility Functions
# ------------------
def calculate_accuracy(predictions, y_true):
    """Calculate classification accuracy"""
    # Get class predictions from probabilities
    predicted_classes = np.argmax(predictions, axis=1)

    # Convert one-hot encoded labels to sparse if needed
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)

    # Calculate accuracy
    accuracy = np.mean(predicted_classes == y_true)
    return accuracy

# ======================
# MAIN TRAINING SCRIPT
# ======================

print("=" * 60)
print("COMPLETE NEURAL NETWORK FROM SCRATCH")
print("=" * 60)

# Generate spiral dataset
print("\n1. Generating spiral dataset...")
X, y = spiral_data(samples=100, classes=3)
print(f"   Dataset shape: {X.shape}")
print(f"   Labels shape: {y.shape}")
print(f"   Classes: {np.unique(y)}")

# Create network architecture
print("\n2. Building neural network...")
dense1 = Layer_Dense(2, 3)                    # Input layer: 2 features -> 3 neurons
activation1 = Activation_ReLU()               # ReLU activation
dense2 = Layer_Dense(3, 3)                    # Output layer: 3 neurons -> 3 classes
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()  # Softmax + Loss

print("   Architecture:")
print("   Input (2) -> Dense(3) -> ReLU -> Dense(3) -> Softmax -> CrossEntropy")

# Create optimizer
optimizer = Optimizer_SGD(learning_rate=0.01)

# Training parameters
epochs = 53000
print_every = 1000

print(f"\n3. Training for {epochs} epochs...")
print(f"   Learning rate: {optimizer.learning_rate}")

# Training loop
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)

    # Calculate loss
    loss = loss_activation.forward(dense2.output, y)
    train_losses.append(loss)

    # Calculate accuracy
    predictions = loss_activation.output
    accuracy = calculate_accuracy(predictions, y)
    train_accuracies.append(accuracy)

    # Print progress
    if epoch % print_every == 0:
        print(f'   Epoch {epoch:5d} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update parameters
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

# Final results
final_loss = train_losses[-1]
final_accuracy = train_accuracies[-1]

print(f"\n4. Training completed!")
print(f"   Final Loss: {final_loss:.4f}")
print(f"   Final Accuracy: {final_accuracy:.4f}")

# Show sample predictions
print(f"\n5. Sample predictions (first 5 samples):")
print("   True labels:", y[:5])
print("   Predictions:", np.argmax(predictions[:5], axis=1))
print("   Probabilities:")
for i in range(5):
    probs = predictions[i]
    print(f"   Sample {i}: {probs}")

# Visualize training progress
print(f"\n6. Plotting training progress...")

plt.figure(figsize=(15, 5))

# Plot loss
plt.subplot(1, 3, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Plot accuracy
plt.subplot(1, 3, 2)
plt.plot(train_accuracies)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

# Plot decision boundary
plt.subplot(1, 3, 3)
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Make predictions on mesh
mesh_points = np.c_[xx.ravel(), yy.ravel()]
dense1.forward(mesh_points)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss_activation.activation.forward(dense2.output)
mesh_predictions = np.argmax(loss_activation.activation.output, axis=1)

# Plot decision boundary
mesh_predictions = mesh_predictions.reshape(xx.shape)
plt.contourf(xx, yy, mesh_predictions, alpha=0.8, cmap=plt.cm.RdYlBu)

# Plot data points
colors = ['red', 'blue', 'green']
for i in range(3):
    plt.scatter(X[y == i, 0], X[y == i, 1], c=colors[i], label=f'Class {i}')

plt.title('Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()

print(f"\n7. Visualization complete!")
print("=" * 60)