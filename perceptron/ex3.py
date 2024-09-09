import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Initial inputs, weights, bias, and learning rate
x1, x2 = 0.7, -0.6
weights = np.array([0.5, 0.5])
bias = 0.5
alpha = 0.1
target_output = 1

# Forward pass
input_sum = np.dot([x1, x2], weights) + bias
predicted_output = sigmoid(input_sum)

# Delta Rule (Weight Update)
error = target_output - predicted_output
delta = error * sigmoid_derivative(input_sum)

# Update weights and bias
weights += alpha * delta * np.array([x1, x2])
bias += alpha * delta

print(f"Updated Weights: {weights}")
print(f"Updated Bias: {bias}")
