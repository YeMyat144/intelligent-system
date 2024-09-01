import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv('/mnt/data/bmi.csv')  # Adjust the path if necessary

# Features and labels
X = data[['Height', 'Weight']].values
y = data['BMI'].values

# Initialize weights and bias
weights = np.random.rand(2)
bias = np.random.rand()
alpha = 0.01

def linear_activation(x):
    return x

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
