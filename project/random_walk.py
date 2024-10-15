import random

# Function to calculate BMI
def calculate_bmi(weight, height):
    height_m = height / 100  
    bmi = weight / (height_m ** 2)
    return bmi

# Function to calculate BMR based on Mifflin-St Jeor Equation
def calculate_bmr(weight, height, age, gender):
    if gender == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    return bmr

# Function to adjust BMR based on activity level
def adjust_bmr_for_activity(bmr, activity_level):
    activity_factors = {
        'sedentary': 1.2,
        'lightly active': 1.375,
        'moderately active': 1.55,
        'very active': 1.725,
        'super active': 1.9
    }
    return bmr * activity_factors.get(activity_level.lower(), 1.2)

# Function for Random Walk to match calorie target
def random_walk_calories(food_items, target_calories, max_steps=1000):
    current_solution = random.sample(food_items, len(food_items))
    for _ in range(max_steps):
        total_calories = sum(item['calories'] for item in current_solution)
        if abs(target_calories - total_calories) < 50:
            break
        # Randomly swap a food item for a new one to adjust calories
        current_solution[random.randint(0, len(current_solution)-1)] = random.choice(food_items)
    return current_solution

# Example food items with calories (breakfast, lunch, dinner)
food_items = [
    {"name": "Apple", "calories": 52},
    {"name": "Banana", "calories": 89},
    {"name": "Chicken Breast", "calories": 165},
    {"name": "Rice", "calories": 130},
    {"name": "Salad", "calories": 33},
    {"name": "Egg", "calories": 78},
    {"name": "Yogurt", "calories": 59},
    {"name": "Bread", "calories": 80},
    {"name": "Pasta", "calories": 131},
    {"name": "Steak", "calories": 271},
    {"name": "Fish", "calories": 206},
    {"name": "Oatmeal", "calories": 158},
    {"name": "Smoothie", "calories": 200},
]

# Function to divide the meal plan into breakfast, lunch, and dinner
def split_meal_plan(meal_plan):
    random.shuffle(meal_plan)
    breakfast = meal_plan[:3]  
    lunch = meal_plan[3:6]     
    dinner = meal_plan[6:9]    
    return breakfast, lunch, dinner

weight = int(input("Enter your weight in kilograms (kg): "))
height = int(input("Enter your height in centimeters (cm): "))
age = int(input("Enter your age in years: "))
gender = str(input("Enter your gender (male/female): ")).lower()
activity_level = str(input("Enter your daily activity level (sedentary, lightly active, moderately active, very active, super active): ")).lower()

# Calculate BMI and BMR
bmi = calculate_bmi(weight, height)
bmr = calculate_bmr(weight, height, age, gender)
adjusted_bmr = adjust_bmr_for_activity(bmr, activity_level)

# Example target calories based on weight goals (could be user input too)
target_calories = adjusted_bmr - 500  # To lose weight, subtract 500 from BMR

# Generate a meal plan that matches target calories using Random Walk
best_meal_plan = random_walk_calories(food_items, target_calories)

# Split the meal plan into breakfast, lunch, and dinner
breakfast, lunch, dinner = split_meal_plan(best_meal_plan)

# Output the results
print(f"\nUser BMI: {bmi:.2f}")
print(f"User BMR: {bmr:.2f} kcal/day")
print(f"Adjusted BMR based on activity: {adjusted_bmr:.2f} kcal/day")
print(f"Target Calories for weight loss: {target_calories:.2f} kcal/day")

print("\nSuggested Meal Plan:")
print("\nBreakfast:")
for food in breakfast:
    print(f" - {food['name']} ({food['calories']} kcal)")

print("\nLunch:")
for food in lunch:
    print(f" - {food['name']} ({food['calories']} kcal)")

print("\nDinner:")
for food in dinner:
    print(f" - {food['name']} ({food['calories']} kcal)")
