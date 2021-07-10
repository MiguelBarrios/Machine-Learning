
import numpy as np

def print_V_e():
	states = ["Olympus", "Delphi", "Delos", "Dodoni"]
	for i in range(4):
		print(states[i] + " : V = " + str(V[i]) + " e = " + str(e[i]))

def update_TD_lamda(cur_state, action, r, next_state, s_primes):
	# decay the traces
	for s_prime in s_primes:
		e[s_prime] = gamma * lam * e[s_prime]

	e[cur_state] = e[cur_state] + 1
	delta = r + gamma * V[next_state] - V[cur_state]
	for s in range(num_states):
		V[s] = V[s] + alpha * delta * e[s]

alpha, gamma, lam = 0.05, 0.9, 0.6
num_states = 4

Olympus, Delphi, Delos, Dodoni = 0, 1, 2 ,3,
fly, walk, horse = 0, 1, 2

availableNextStates = [[Olympus, Delphi, Dodoni],
[Olympus, Delphi,Delos, Dodoni],
[Delos, Delphi, Dodoni], 
[Dodoni, Olympus, Delphi]]

V = np.zeros(num_states)
e = np.zeros(num_states)

# Episode 1: Olympus, walk, 2, Dodoni, fly, 2, Olympus, fly, -1, Olympus]
print("First Episode")
update_TD_lamda(Olympus, walk, 2, Dodoni, availableNextStates[Olympus])
update_TD_lamda(Dodoni,fly, 2,Olympus, availableNextStates[Dodoni])
update_TD_lamda(Olympus,fly, -1, Olympus, availableNextStates[Olympus])
print_V_e()

e = np.zeros(num_states)
print("\nSecond Episode")
# Episode 2: Olympus, fly, 2, Delphi, fly, 4, Delos
update_TD_lamda(Olympus, fly, 2, Delphi,availableNextStates[Olympus])
update_TD_lamda(Delphi, fly, 4, Delos,availableNextStates[Olympus])
print_V_e()


