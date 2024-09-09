import pandas as pd
import numpy as np

# Load the dataset (assuming bmi.csv is properly formatted with Height, Weight, and BMI columns)
data = pd.read_csv(r'C:\Users\Ye Myat Moe\Documents\sp\intelligent_system\perceptron\bmi.csv', header=None)
data.columns = ['Gender', 'Height', 'Weight', 'BMI']  
data = data[['Height', 'Weight', 'BMI']]

# Normalize the data
data['Height'] = (data['Height'] - data['Height'].min()) / (data['Height'].max() - data['Height'].min())
data['Weight'] = (data['Weight'] - data['Weight'].min()) / (data['Weight'].max() - data['Weight'].min())
data['BMI'] = (data['BMI'] - data['BMI'].min()) / (data['BMI'].max() - data['BMI'].min())

# Initialize weights and bias
weights = np.random.rand(2)
bias = np.random.rand()
alpha = 0.1

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def perceptron_train_bmi(data, weights, bias, alpha, epochs=1000):
    for epoch in range(epochs):
        for i in range(len(data)):
            inputs = np.array([data.iloc[i]['Height'], data.iloc[i]['Weight']])
            target_output = data.iloc[i]['BMI']
            input_sum = np.dot(inputs, weights) + bias
            predicted_output = sigmoid(input_sum)
            error = target_output - predicted_output
            
            # Update weights and bias
            delta = error * predicted_output * (1 - predicted_output)
            weights += alpha * delta * inputs
            bias += alpha * delta
    return weights, bias

# Train the perceptron
trained_weights, trained_bias = perceptron_train_bmi(data, weights, bias, alpha)
print(f"Trained Weights: {trained_weights}")
print(f"Trained Bias: {trained_bias}")

# Test prediction on a new sample
def predict_bmi(height, weight, weights, bias):
    inputs = np.array([height, weight])
    return sigmoid(np.dot(inputs, weights) + bias)

# Example prediction
height = int(input("Enter height: "))
weight = int(input("Enter weight: "))
predicted_bmi = predict_bmi(height, weight, trained_weights, trained_bias)
print(f"Predicted BMI: {predicted_bmi}")
