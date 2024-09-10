import numpy as np
import pandas as pd

# Load dataset without header and manually assign column names
data = pd.read_csv(r'C:\Users\Ye Myat Moe\Documents\sp\intelligent_system\mlp_bmi\bmi.csv', header=None)
data.columns = ['Gender', 'Height', 'Weight', 'BMI']  # Manually assigning the column names

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

# Initialize weights and biases for a 3-x-1 topology
input_layer_neurons = X.shape[1]  # 3 inputs
hidden_layer_neurons = 5  # Arbitrary hidden layer size
output_neurons = 1  # 1 output (BMI)

# Random weight initialization
w_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
b_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
w_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
b_output = np.random.uniform(size=(1, output_neurons))

# Learning rate and epochs
learning_rate = 0.01
epochs = 10000

# Training the MLP
for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(X, w_input_hidden) + b_hidden
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output, w_hidden_output) + b_output
    predicted_output = sigmoid(output_layer_activation)
    
    # Compute the error
    error = y - predicted_output
    
    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(w_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    w_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    b_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    w_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    b_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
    
    # Print the error for every 1000 epochs
    if epoch % 1000 == 0:
        mse = np.mean(np.square(error))
        print(f"Epoch {epoch}, Mean Squared Error: {mse}")

# Output the final weights and biases
print("Training complete.")
print("Final weights (input to hidden):", w_input_hidden)
print("Final weights (hidden to output):", w_hidden_output)
