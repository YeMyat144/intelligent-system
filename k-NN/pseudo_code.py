import numpy as np
import pandas as pd
import random
from collections import Counter

def generate_random_dataset(size):
    data = {
        "Age (yrs)": [random.randint(18, 65) for _ in range(size)],   
        "Salary ($)": [round(random.uniform(2000, 15000), 2) for _ in range(size)],  
        "Hours/week": [round(random.uniform(5, 50), 1) for _ in range(size)]   
    }
    return pd.DataFrame(data)  

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))  

def k_nearest_neighbors(data, test_point, k):
    distances = []
    for i in range(len(data)):
        
        distances.append((euclidean_distance(data.iloc[i], test_point), i))
    #distances.sort(key=lambda x: x[0])  Sort distances in ascending order
    distances.sort(key=lambda x: x[0], reverse=True)  
    nearest_neighbors_indices = [distances[i][1] for i in range(k)]  
    nearest_neighbors = data.iloc[nearest_neighbors_indices].copy()  
    nearest_neighbors['Distance'] = [distances[i][0] for i in range(k)]  
    return nearest_neighbors  

def main():
    while True:
        size = int(input("Enter the size of the random dataset (Age, Salary, Hours/week) <max 50>: "))
        if 0 < size <= 50:
            break
        print("Size should be less than or equal to 50 and greater than 0.")

    dataset = generate_random_dataset(size)  
    print("Generated Dataset:")
    print(dataset.to_string(index=False))   

    test_age = int(input("Enter the Age (yrs) for the test data: "))
    test_salary = float(input("Enter the Salary ($) for the test data: "))
    test_hours = float(input("Enter the Hours/week for the test data: "))
    test_point = np.array([test_age, test_salary, test_hours]) 
    
    k = int(input("Enter the value of k: ")) 

    neighbors = k_nearest_neighbors(dataset, test_point, k)   
    
    print(f"\nTop {k} nearest neighbors:")
    print(neighbors.to_string(index=False))  

if __name__ == "__main__":
    main()
