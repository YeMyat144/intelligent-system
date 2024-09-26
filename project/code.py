import random

# Sample food dataset (food item: calorie)
food_items = {
    "apple": 95, "banana": 105, "bread": 80, "cheese": 120, "chicken": 165,
    "egg": 70, "fish": 200, "milk": 150, "orange": 62, "rice": 206
}

# Calculate BMI and BMR
def calculate_bmi(weight, height):
    return weight / (height ** 2)

def calculate_bmr(weight, height, age, gender):
    if gender == "male":
        return 88.36 + (13.4 * weight) + (4.8 * height * 100) - (5.7 * age)
    else:
        return 447.6 + (9.2 * weight) + (3.1 * height * 100) - (4.3 * age)

# Define fitness function
def fitness(meal_plan, target_calories):
    total_calories = sum(food_items[item] for item in meal_plan)
    return abs(target_calories - total_calories)

# Genetic Algorithm
def genetic_algorithm(target_calories, population_size=10, generations=20):
    population = [random.sample(list(food_items.keys()), 3) for _ in range(population_size)]
    
    for generation in range(generations):
        population.sort(key=lambda meal: fitness(meal, target_calories))
        
        # Selection
        selected = population[:5]
        
        # Crossover
        for _ in range(5):
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            crossover_point = random.randint(1, len(parent1) - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            selected.append(child)
        
        # Mutation
        for meal in selected:
            if random.random() < 0.1:
                meal[random.randint(0, len(meal) - 1)] = random.choice(list(food_items.keys()))
        
        population = selected
    
    # Best solution
    best_meal = population[0]
    best_calories = sum(food_items[item] for item in best_meal)
    return best_meal, best_calories

# Random Walk Algorithm
def random_walk(target_calories, max_steps=100):
    current_meal = random.sample(list(food_items.keys()), 3)
    
    for _ in range(max_steps):
        total_calories = sum(food_items[item] for item in current_meal)
        
        if total_calories == target_calories:
            return current_meal, total_calories
        
        random_food = random.choice(list(food_items.keys()))
        replace_index = random.randint(0, len(current_meal) - 1)
        current_meal[replace_index] = random_food
    
    return current_meal, sum(food_items[item] for item in current_meal)

# Example usage
weight = 70  # kg
height = 1.75  # meters
age = 25
gender = "male"

bmi = calculate_bmi(weight, height)
bmr = calculate_bmr(weight, height, age, gender)
target_calories = bmr * 0.8  # Adjust target calories as needed

print("Genetic Algorithm Result:")
meal_ga, calories_ga = genetic_algorithm(target_calories)
print(f"Meal Plan: {meal_ga}, Total Calories: {calories_ga}")

print("\nRandom Walk Algorithm Result:")
meal_rw, calories_rw = random_walk(target_calories)
print(f"Meal Plan: {meal_rw}, Total Calories: {calories_rw}")
