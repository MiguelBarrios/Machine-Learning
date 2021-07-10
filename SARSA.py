import numpy as np
import random

def print_q_table():
	states = ["Olympus", "Delphi", "Delos", "Dodoni"]
	actions = ["Fly", "Walk", "Horse"]
	for state in range(num_states):
	    for action in range(num_actions):
	        print("(" + states[state] + ", " + actions[action] + ") = " + str(q_table[state, action]))

def SARSA_update(state, action, reward, next_state, next_action):
	q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state,action])

alpha = 0.05
gamma = 0.9

num_states = 4
num_actions = 3

Olympus, Delphi, Delos, Dodoni = 0, 1, 2 ,3,
fly, walk, horse = 0, 1, 2

q_table = np.zeros((num_states,num_actions))

# Episode 1: Olympus, walk, 2, Dodoni, fly, 2, Olympus, fly, -1, Olympus
print("#### Episode 1 ####")
SARSA_update(Olympus, walk, 2, Dodoni, fly)
SARSA_update(Dodoni, fly, 2, Olympus, fly)
SARSA_update(Olympus, fly, -1, Olympus, fly)

print_q_table()
print("\n#### Episode 2 ####")
# Episode 2: Olympus, fly, 2, Delphi, fly, 4, Delos
SARSA_update(Olympus, fly, 2, Delphi, fly)
SARSA_update(Delphi, fly, 4, Delos, fly)
print_q_table()
