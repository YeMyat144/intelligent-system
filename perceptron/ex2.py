import numpy as np

# Perceptron Parameters
w1, w2 = 0.3, -0.1
theta = 0.2
alpha = 0.1  # Learning rate

def perceptron_training(inputs, outputs, w1, w2, theta, alpha, epochs=10):
    for epoch in range(epochs):
        for input, output in zip(inputs, outputs):
            x1, x2 = input
            weighted_sum = w1 * x1 + w2 * x2
            y = step_function(weighted_sum - theta)
            error = output - y
            w1 += alpha * error * x1
            w2 += alpha * error * x2
            theta += alpha * error * -1
        print(f"Epoch {epoch+1}: w1 = {w1}, w2 = {w2}, theta = {theta}")
    return w1, w2, theta

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
outputs = [0, 1, 1, 1]  # OR function
w1, w2, theta = perceptron_training(inputs, outputs, w1, w2, theta, alpha)
