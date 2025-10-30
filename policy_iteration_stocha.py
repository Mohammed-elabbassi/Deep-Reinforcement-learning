import numpy as np

theta = 0.0001
gamma = 0.9
reword_step = -1

rows, cols = 4, 4
V = np.zeros((rows, cols))

goal, trap = (3, 3), (0, 3)
V[goal], V[trap] = 0, 0

actions = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1)
}

P = {
    "up": {"up": 0.7, "left": 0.1, "right": 0.1, "down": 0.1},
    "down": {"down": 0.7, "left": 0.1, "right": 0.1, "up": 0.1},
    "left": {"left": 0.7, "up": 0.1, "down": 0.1, "right": 0.1},
    "right": {"right": 0.7, "up": 0.1, "down": 0.1, "left": 0.1},
}

def is_valid(r, c):
    return 0 <= r < rows and 0 <= c < cols

def next_move(r, c, action):
    dr, dc = actions[action]
    new_r, new_c = r + dr, c + dc
    if not is_valid(new_r, new_c):
        new_r, new_c = r, c
    return new_r, new_c

iteration = 0

while True:
    delta = 0
    V_new = np.copy(V)
    for r in range(rows):
        for c in range(cols):
            if (r, c) in [goal, trap]:
                continue

            values = []
            for a in actions:
                q = 0
                for move, prob in P[a].items():
                    new_r, new_c = next_move(r, c, move)

                    if (new_r, new_c) == goal:
                        reword = 10
                    elif (new_r, new_c) == trap:
                        reword = -10
                    else:
                        reword = reword_step

                    q += prob * (reword + gamma * V[new_r, new_c])
                values.append(q)

            # uniform random policy = average over all actions
            V_new[r, c] = np.mean(values)

            delta = max(delta, abs(V_new[r, c] - V[r, c]))

    V = V_new
    iteration += 1
    print(f"Iteration {iteration}:\n{V}\n")

    if delta < theta:
        break

print("Converged in {} iterations".format(iteration))
print("Optimal Value Function:")
print(V)

policy = np.full((rows, cols), '', dtype=object)  # to store best action per state

for r in range(rows):
    for c in range(cols):
        if (r, c) == goal :
            policy[r, c] = 'Goal'
            continue
        if (r, c) == trap :
            policy[r, c] = 'Trap'
            continue

        q_values = {}
        for a in actions:
            q = 0
            for move, prob in P[a].items():
                new_r, new_c = next_move(r, c, move)
                if (new_r, new_c) == goal:
                    reword = 10
                elif (new_r, new_c) == trap:
                    reword = -10
                else:
                    reword = reword_step
                q += prob * (reword + gamma * V[new_r, new_c])
            q_values[a] = q
        # Choose the best action (max Q)
        best_action = max(q_values, key=q_values.get)
        policy[r, c] = best_action

print("Optimal Policy:")
arrow_map = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
    "Goal": "G",
    "Trap": "H"
}

for r in range(rows):
    print(" ".join(arrow_map[a] for a in policy[r]))