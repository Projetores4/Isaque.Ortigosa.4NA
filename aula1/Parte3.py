import numpy as np

# Define o ambiente
n_states = 6
n_actions = 4
Q = np.zeros((n_states, n_actions))

# Define os parâmetros de aprendizado
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.1

# Define as recompensas
rewards = np.array([[-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, 100]])

# Treina o agente
for i in range(1000):
    state = np.random.randint(0, n_states)
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(0, n_actions)
    else:
        action = np.argmax(Q[state])
    next_state = np.random.randint(0, n_states)
    Q[state, action] = Q[state, action] + learning_rate * (rewards[state, action] + discount_factor * np.max(Q[next_state]) - Q[state, action])

# Imprime a tabela Q
print(f"Tabela Q: {Q}")