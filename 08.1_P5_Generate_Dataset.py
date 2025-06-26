import numpy as np
import matplotlib.pyplot as plt

# Set a random seed so results are reproducible
np.random.seed(0)

# --------- Custom Data Generation Function ---------

def create_data(points, classes):
    """
    Generates a 2D spiral dataset for classification tasks.

    Parameters:
    - points: number of points (samples) per class
    - classes: total number of classes (spirals)

    Returns:
    - X: array of shape (points * classes, 2), containing 2D coordinates
    - y: array of shape (points * classes,), containing class labels
    """
    # Create empty arrays for features and labels
    X = np.zeros((points * classes, 2))              # Features: 2D coordinates
    y = np.zeros(points * classes, dtype='uint8')    # Labels: class index for each point

    for class_number in range(classes):
        # Create a slice (range of indices) for the current class
        ix = range(points * class_number, points * (class_number + 1))

        # Generate radius (r) linearly spaced from 0 to 1
        r = np.linspace(0.0, 1, points)

        # Generate theta (t) linearly spaced, then add noise for spiral shape
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) 
        t += np.random.randn(points) * 0.2  # Add random noise to make it less uniform

        # Convert polar coordinates (r, t) to Cartesian coordinates (x, y)
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]

        # Assign the class number as the label
        y[ix] = class_number

    return X, y

# --------- Generate and Visualize the Data ---------

print("here")  # Just a checkpoint message

# Generate 300 data points (100 per class) for 3 spiral-shaped classes
X, y = create_data(100, 3)

# Plot without class colors (all points are the same color)
plt.scatter(X[:, 0], X[:, 1])
plt.title("Spiral Dataset (No Labels)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# Plot with class-specific colors
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")  # "brg" gives distinct colors for classes
plt.title("Spiral Dataset (Colored by Class)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
