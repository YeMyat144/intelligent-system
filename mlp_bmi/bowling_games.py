def bowling_game(scores):
    # Calculate minimum score
    min_score = 0
    spare = False
    for i in range(9):
        min_score += scores[i]
        spare = scores[i] == 10
    min_score += scores[9]
    
    if scores[9] > 20:
        min_score += 10 * (1 if spare else 0)
    
    # Calculate maximum score
    max_score = 0
    strike = 0
    for i in range(9):
        max_score += scores[i] * (1 + strike)
        strike = 0 if scores[i] < 10 else min(2, strike + 1)
    
    max_score += min(scores[9], 10) * (1 + strike)
    if scores[9] > 10:
        max_score += min(10, scores[9] - 10) * (1 + (1 if strike in (1, 2) else 0)) \
                     + max(scores[9] - 20, 0)
    
    return min_score, max_score

# Input scores as a list
# Input scores as a space-separated string and convert them to integers
scores = list(map(int, input().split()))

# Ensure that exactly 10 numbers are provided
if len(scores) != 10:
    raise ValueError("You must enter exactly 10 scores")

# Calculate and print the result
min_score, max_score = bowling_game(scores)
print(min_score, max_score)
