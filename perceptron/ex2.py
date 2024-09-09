import numpy as np

# OR truth table inputs and outputs
training_data = [
    (np.array([0, 0]), 0),
    (np.array([0, 1]), 1),
    (np.array([1, 0]), 1),
    (np.array([1, 1]), 1),
]

# Initial weights, threshold, and learning rate
w1, w2 = 0.3, -0.1
theta = 0.2
alpha = 0.1

weights = np.array([w1, w2])

def perceptron_train(data, weights, threshold, alpha):
    for _ in range(10):  # Train for 10 epochs
        for inputs, expected_output in data:
            weighted_sum = np.dot(inputs, weights)
            output = 1 if weighted_sum >= threshold else 0
            error = expected_output - output
            # Update weights using Perceptron rule
            weights += alpha * error * inputs
            print(f"Updated Weights: {weights}")
    return weights

trained_weights = perceptron_train(training_data, weights, theta, alpha)
print(f"Final Trained Weights: {trained_weights}")
