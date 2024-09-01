import numpy as np
import pandas as pd

# Step Function (for Exercise 1 and 2)
def step_function(x):
    return 1 if x >= 0 else 0

# Sigmoid Function and its derivative (for Exercise 3)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Linear Activation Function (for Exercise 4)
def linear_activation(x):
    return x

# Exercise 1: Create two input AND and OR functions using a Perceptron
def exercise_1():
    def perceptron(x1, x2, w1, w2, bias):
        weighted_sum = w1 * x1 + w2 * x2 + bias
        return step_function(weighted_sum)

    print("\n-- AND Function --")
    w1_and = float(input("Enter weight 1 for AND: "))
    w2_and = float(input("Enter weight 2 for AND: "))
    bias_and = float(input("Enter bias (threshold) for AND: "))

    print("AND Function:")
    for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        print(f"Input: ({x1}, {x2}) => Output: {perceptron(x1, x2, w1_and, w2_and, bias_and)}")

    print("\n-- OR Function --")
    w1_or = float(input("Enter weight 1 for OR: "))
    w2_or = float(input("Enter weight 2 for OR: "))
    bias_or = float(input("Enter bias (threshold) for OR: "))

    print("\nOR Function:")
    for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        print(f"Input: ({x1}, {x2}) => Output: {perceptron(x1, x2, w1_or, w2_or, bias_or)}")

# Exercise 2: Train a perceptron for getting 2-input OR function
def exercise_2():
    w1 = float(input("Enter initial weight 1 for OR: "))
    w2 = float(input("Enter initial weight 2 for OR: "))
    theta = float(input("Enter initial threshold value (theta): "))
    alpha = float(input("Enter learning rate (alpha): "))

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
    perceptron_training(inputs, outputs, w1, w2, theta, alpha)

# Exercise 3: Perceptron with Sigmoid Activation and Delta Rule
def exercise_3():
    weights = np.array([float(input("Enter initial weight 1: ")), 
                        float(input("Enter initial weight 2: "))])
    bias = float(input("Enter initial bias (threshold): "))
    alpha = float(input("Enter learning rate (alpha): "))

    x1 = float(input("Enter input x1: "))
    x2 = float(input("Enter input x2: "))
    desired_output = float(input("Enter desired output: "))

    for epoch in range(10):  # Run for 10 epochs
        weighted_sum = np.dot(weights, np.array([x1, x2])) + bias
        output = sigmoid(weighted_sum)
        error = desired_output - output
        delta = error * sigmoid_derivative(weighted_sum)

        # Update weights and bias
        weights += alpha * delta * np.array([x1, x2])
        bias += alpha * delta * 1

        print(f"Epoch {epoch+1}: weights = {weights}, bias = {bias}")

# Exercise 4: Predict BMI using a Perceptron
def exercise_4():
    # Load dataset (adjust path as necessary)
    data = pd.read_csv('/mnt/data/bmi.csv')

    # Features and labels
    X = data[['Height', 'Weight']].values
    y = data['BMI'].values

    # Initialize weights and bias
    weights = np.random.rand(2)
    bias = np.random.rand()
    alpha = float(input("Enter learning rate (alpha): "))

    # Training process
    for epoch in range(100):  # Run for 100 epochs
        for i in range(len(X)):
            x_i = X[i]
            y_i = y[i]

            # Calculate prediction
            weighted_sum = np.dot(weights, x_i) + bias
            output = linear_activation(weighted_sum)

            # Calculate error
            error = y_i - output

            # Update weights and bias
            weights += alpha * error * x_i
            bias += alpha * error * 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: weights = {weights}, bias = {bias}")

    # Final weights and bias
    print(f"Final weights = {weights}, bias = {bias}")

# Running all exercises sequentially with input prompts
def main():
    print("Exercise 1:")
    exercise_1()
    print("\nExercise 2:")
    exercise_2()
    print("\nExercise 3:")
    exercise_3()
    print("\nExercise 4:")
    exercise_4()

if __name__ == "__main__":
    main()
