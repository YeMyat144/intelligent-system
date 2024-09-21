from collections import deque

goal_state = "123456780"

moves = {
    0: [1, 3],      # Top-left corner
    1: [0, 2, 4],   # Top-middle
    2: [1, 5],      # Top-right corner
    3: [0, 4, 6],   # Middle-left
    4: [1, 3, 5, 7],# Center
    5: [2, 4, 8],   # Middle-right
    6: [3, 7],      # Bottom-left corner
    7: [4, 6, 8],   # Bottom-middle
    8: [5, 7]       # Bottom-right corner
}

def solve_puzzle(initial_state):
    queue = deque([(initial_state, initial_state.index("0"), 0)])  
    visited = set([initial_state])

    while queue:
        current_state, zero_index, depth = queue.popleft()

        if current_state == goal_state:
            return depth  

        for move in moves[zero_index]:
            new_state = list(current_state)
            new_state[zero_index], new_state[move] = new_state[move], new_state[zero_index]
            new_state = ''.join(new_state)

            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, move, depth + 1))

initial_state = ""
for _ in range(3):
    row = input().strip().split()
    initial_state += ''.join(row)

print(solve_puzzle(initial_state))
