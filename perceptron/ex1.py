import numpy as np

def step_function(x):
    return 1 if x >= 0 else 0

def perceptron(x1, x2, w1, w2, bias):
    weighted_sum = w1 * x1 + w2 * x2 + bias
    return step_function(weighted_sum)

# AND function
weights_and = [0.5, 0.5]
bias_and = -0.7  # Threshold
print("AND Function:")
for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f"Input: ({x1}, {x2}) => Output: {perceptron(x1, x2, weights_and[0], weights_and[1], bias_and)}")

# OR function
weights_or = [0.5, 0.5]
bias_or = -0.2  # Threshold
print("\nOR Function:")
for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f"Input: ({x1}, {x2}) => Output: {perceptron(x1, x2, weights_or[0], weights_or[1], bias_or)}")
