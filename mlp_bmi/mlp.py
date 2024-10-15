import pandas as pd
import numpy as np
from numpy import random

learning_rate = 0.1
epochs = 1000
df = pd.read_csv(r'C:\Users\Ye Myat Moe\Documents\sp\intelligent_system\mlp_bmi\bmi.csv', header=None, names=["Gender", "Height", "Weight", "Index"])
print(df.head())

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
x0 = df['Gender'].values
x1 = df['Height'].values
x2 = df['Weight'].values
y = df['Index'].values

def normalize_data(data):
    return (data - data.mean()) / data.std()

def sigmoid(x):
    # Clip values to prevent overflow in exp
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def prediction(gender, height, weight, weights, bias):
    weighted_sum = gender * weights[0] + height * weights[1] + weight * weights[2] - bias
    sigmoid_output = sigmoid(weighted_sum)
    return sigmoid_output * 5

def fit(x0, x1, x2, y, epochs):
    X = np.column_stack((x0, x1, x2))
    weights = [random.rand(3) for _ in range(4)]
    biases = random.rand(4)
    history = []
    learning_rate = 0.01

    for epoch in range(epochs):
        errors = []

        for g, h, w, ytrain in zip(x0, x1, x2, y):
            # Use different variable names inside loop to avoid conflict
            x1_pred = prediction(g, h, w, weights[0], biases[0])
            x2_pred = prediction(g, h, w, weights[1], biases[1])
            x3_pred = prediction(g, h, w, weights[2], biases[2])

            ypred = prediction(x1_pred, x2_pred, x3_pred, weights[3], biases[3])
            error = ytrain - ypred
            # y_preditcion equation
            error_gy = ypred * (1 - ypred) * error

            w41 = learning_rate * x1_pred * error_gy 
            w42 = learning_rate * x2_pred * error_gy
            w43 = learning_rate * x3_pred * error_gy
            bias4 = learning_rate * -1 * error_gy

            error_gx1 = x1_pred * (1 - x1_pred) * error_gy * weights[3][0]
            error_gx2 = x2_pred * (1 - x2_pred) * error_gy * weights[3][1]
            error_gx3 = x3_pred * (1 - x3_pred) * error_gy * weights[3][2]

            w11 = learning_rate * g * error_gx1
            w21 = learning_rate * h * error_gx1
            w31 = learning_rate * w * error_gx1
            bias1 = learning_rate * -1 * error_gx1

            w12 = learning_rate * g * error_gx2
            w22 = learning_rate * h * error_gx2
            w32 = learning_rate * w * error_gx2
            bias2 = learning_rate * -1 * error_gx2

            w13 = learning_rate * g * error_gx3
            w23 = learning_rate * h * error_gx3
            w33 = learning_rate * w * error_gx3
            bias3 = learning_rate * -1 * error_gx3

            weights[0][0] += w11
            weights[0][1] += w12
            weights[0][2] += w13

            weights[1][0] += w21
            weights[1][1] += w22
            weights[1][2] += w23

            weights[2][0] += w31
            weights[2][1] += w32
            weights[2][2] += w33

            weights[3][0] += w41
            weights[3][1] += w42
            weights[3][2] += w43

            errors.append(error)

        if epoch % 100 == 0:
            mse = np.mean(np.square(errors))
            print(f"Epoch {epoch + 1}, MSE = {mse}")

    print(f"Converged at iteration {epoch + 1}. Last MSE = {mse}")
    return weights, biases

norm_x0 = normalize_data(x0) # gender
norm_x1 = normalize_data(x1) # height
norm_x2 = normalize_data(x2) # weight

weights, biases = fit(norm_x0, norm_x1, norm_x2, y, epochs)

# Assuming you have new values for height and weight
new_height = int(input("Enter new height: "))
new_weight = int(input("Enter new weight: "))

# Normalize the new data using the same normalization function
normalized_new_height = (new_height - x1.mean()) / x1.std()
normalized_new_weight = (new_weight - x2.mean()) / x2.std()

# Combine the normalized values into a single array
new_data = np.array([1, normalized_new_height, normalized_new_weight])

# Make a prediction using the trained weights and biases
print()
print(1, new_height, new_weight)
print(weights)
prediction = np.dot(new_data, weights[3]) + biases[3]

print()
print("Predicted Probability:", prediction)