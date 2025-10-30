
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


policy = np.random.choice(list(actions.keys()), size=(rows, cols))
policy[goal] = "TERM"
policy[trap] = "TERM"


iteration = 0

while True:
   
    while True:
        delta = 0
        V_new = np.copy(V)
        for r in range(rows):
            for c in range(cols):
                if (r, c) in [goal, trap]:
                    continue
                a = policy[r, c]
                v = 0
                for move, prob in P[a].items():
                    new_r, new_c = next_move(r, c, move)
                    if (new_r, new_c) == goal:
                        reword = 10
                    elif (new_r, new_c) == trap:
                        reword = -10
                    else:
                        reword = reword_step
                    v += prob * (reword + gamma * V[new_r, new_c])
                V_new[r, c] = v
                delta = max(delta, abs(V_new[r, c] - V[r, c]))
        V = V_new
        if delta < theta:
            break

    policy_stable = True
    for r in range(rows):
        for c in range(cols):
            if (r, c) in [goal, trap]:
                continue
            old_action = policy[r, c]
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
            best_action = max(q_values, key=q_values.get)
            policy[r, c] = best_action
            if old_action != best_action:
                policy_stable = False

    iteration += 1
    print(f"Iteration {iteration} Done\nValue Function:\n{np.round(V, 2)}\n")

    if policy_stable:
        break
print(f"Converged in {iteration} iterations")
print(np.round(V, 2))

arrow_map = {"up": "↑", "down": "↓", "left": "←", "right": "→", "TERM": "●"}
for r in range(rows):
    print(" ".join(arrow_map[a] for a in policy[r]))

import gymnasium as gym

rows, cols = 4, 4 
goal, trap = (3, 3), (0, 3) 

GYM_ACTION_TO_IDX = {"left": 0, "down": 1, "right": 2, "up": 3}

optimal_policy_indices = np.zeros((rows, cols), dtype=int)

for r in range(rows):
    for c in range(cols):
        action_name = policy[r, c]
        if action_name in GYM_ACTION_TO_IDX:
            optimal_policy_indices[r, c] = GYM_ACTION_TO_IDX[action_name]
        else:
            
            optimal_policy_indices[r, c] = 0 

print(optimal_policy_indices)


custom_map = [
    "SFFH",  # R=0
    "FFFF",  # R=1
    "FFFF",  # R=2
    "FFFG"   # R=3
]
map_desc = custom_map

print('\n'.join(map_desc))

env = gym.make(
    "FrozenLake-v1", 
    is_slippery=False, 
    desc=map_desc, 
    render_mode="human"
)

def run_episode(env, optimal_policy_indices):
    
    n_cols = env.unwrapped.ncol 
    
    observation, info = env.reset() 
    
    total_reward = 0
    terminated = False
    truncated = False
    step = 0
    
    
    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
    
    while not (terminated or truncated):
    
        r = observation // n_cols 
        c = observation % n_cols
        
        action = optimal_policy_indices[r, c]
        
        
        new_observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step += 1
        
        print(f"Step {step}: State {observation} ({r},{c}) -> Action: {action_names[action]} ({action}) -> New State: {new_observation}, Reward: {reward}")

        observation = new_observation
  
    if terminated and reward == 1.0:
        print(f" Success! Reached the Goal in {step} steps. Total Reward: {total_reward}")
    elif terminated:
        print(f" Failed! Fell into a Hole in {step} steps. Total Reward: {total_reward}")
    elif truncated:
        print(f" Truncated! Max steps reached . Total Reward: {total_reward}")
    
    return total_reward


run_episode(env, optimal_policy_indices)

env.close()