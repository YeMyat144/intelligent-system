import numpy as np

def perceptron(inputs, weights, threshold):
    weighted_sum = np.dot(inputs, weights)
    return 1 if weighted_sum >= threshold else 0

# AND function
weights_and = [0.5, 0.5]
threshold_and = 0.7

# OR function
weights_or = [0.5, 0.5]
threshold_or = 0.3

# Input combinations (truth table)
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

print("AND Perceptron Output:")
for x in inputs:
    print(f"Input: {x}, Output: {perceptron(x, weights_and, threshold_and)}")

print("\nOR Perceptron Output:")
for x in inputs:
    print(f"Input: {x}, Output: {perceptron(x, weights_or, threshold_or)}")
