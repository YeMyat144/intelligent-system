import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Perceptron Parameters
weights = np.array([0.5, -0.5])
bias = 0.5
alpha = 0.1

# Inputs
x1, x2 = 0.7, -0.6
desired_output = 1

for epoch in range(10):  # Run for 10 epochs
    weighted_sum = np.dot(weights, np.array([x1, x2])) + bias
    output = sigmoid(weighted_sum)
    error = desired_output - output
    delta = error * sigmoid_derivative(weighted_sum)
    
    # Update weights and bias
    weights += alpha * delta * np.array([x1, x2])
    bias += alpha * delta * 1
    
    print(f"Epoch {epoch+1}: weights = {weights}, bias = {bias}")
