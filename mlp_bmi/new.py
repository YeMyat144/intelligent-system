import numpy as np
import pandas as pd

# Load dataset without header and manually assign column names
data = pd.read_csv(r'C:\Users\Ye Myat Moe\Documents\sp\intelligent_system\mlp_bmi\bmi.csv', header=None)
data.columns = ['Gender', 'Height', 'Weight', 'BMI'] 

# Preprocessing: Convert Gender to numerical values (assuming Male = 0, Female = 1)
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Normalize the Height and Weight features
data['Height'] = data['Height'] / data['Height'].max()
data['Weight'] = data['Weight'] / data['Weight'].max()

# Features (Gender, Height, Weight) and labels (BMI)
X = data[['Gender', 'Height', 'Weight']].values
y = data['BMI'].values.reshape(-1, 1)

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of the sigmoid function
def sigmoid_derivative(z):
    return z * (1 - z)

# Initialize weights and biases as per provided images
w13, w14, w23, w24, w35, w45 = 0.5, 0.9, 0.4, 1.0, -1.2, 1.1
theta3, theta4, theta5 = 0.8, -0.1, 0.3

# Learning rate and epochs
learning_rate = 0.1  # as per the slides
epochs = 10000

# Training the MLP
for epoch in range(epochs):
    sose = 0  # Initialize Sum of Squared Errors (SOSE) for this epoch
    
    for i in range(X.shape[0]):
        # Forward Propagation
        
        # Calculate hidden layer outputs
        y3 = sigmoid(X[i, 0] * w13 + X[i, 1] * w23 - theta3)
        y4 = sigmoid(X[i, 0] * w14 + X[i, 1] * w24 - theta4)
        
        # Calculate output layer output
        y5 = sigmoid(y3 * w35 + y4 * w45 - theta5)
        
        # Compute the error (difference between actual and predicted output)
        error = y[i] - y5
        
        # Calculate the SOSE for monitoring
        sose += error ** 2
        
        # Backpropagation: Calculate the error gradient for neuron 5 in the output layer
        delta5 = y5 * (1 - y5) * error
        
        # Calculate weight corrections assuming learning rate (alpha) = 0.1
        delta_w35 = learning_rate * y3 * delta5
        delta_w45 = learning_rate * y4 * delta5
        delta_theta5 = learning_rate * (-1) * delta5
        
        # Calculate error gradients for neurons 3 and 4 in the hidden layer
        delta3 = y3 * (1 - y3) * delta5 * w35
        delta4 = y4 * (1 - y4) * delta5 * w45
        
        # Determine the weight corrections for the hidden layer
        delta_w13 = learning_rate * X[i, 0] * delta3
        delta_w23 = learning_rate * X[i, 1] * delta3
        delta_theta3 = learning_rate * (-1) * delta3

        delta_w14 = learning_rate * X[i, 0] * delta4
        delta_w24 = learning_rate * X[i, 1] * delta4
        delta_theta4 = learning_rate * (-1) * delta4
        
        # Update all weights and thresholds
        w13 += delta_w13
        w14 += delta_w14
        w23 += delta_w23
        w24 += delta_w24
        w35 += delta_w35
        w45 += delta_w45

        theta3 += delta_theta3
        theta4 += delta_theta4
        theta5 += delta_theta5

    # Print the Mean Squared Error for every 1000 epochs
    if epoch % 1000 == 0:
        # Mean Squared Error
        mse = sose / X.shape[0] 
        print(f"Epoch {epoch}, Mean Squared Error: {mse}")

    # Early stopping if error is low enough
    if sose < 0.001:
        break

# Output the final weights and biases after training
print("Training complete.")
print(f"Final weights (w13, w14, w23, w24, w35, w45): {w13}, {w14}, {w23}, {w24}, {w35}, {w45}")
print(f"Final biases (theta3, theta4, theta5): {theta3}, {theta4}, {theta5}")