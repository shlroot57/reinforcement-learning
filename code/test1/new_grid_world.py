import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS_LEFT = [4, 0]
A_PRIME_POS_RIGHT = [4, 2]

ACTION_LEFT = 0.7
ACTION_RIGHT = 0.3

B_POS = [0, 3]
B_PRIME_POS = [2, 3]
DISCOUNT = 0.9#折扣系数

# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTIONS_FIGS = ['←', '↑', '→', '↓']

ACTION_PROB = 0.25


def step(state, action):
    if state == B_POS:
        return B_PRIME_POS, 1, 5

    if state == A_POS:
        A_REWARD = np.random.choice([7, 13], 1, p=[0.7, 0.3])[0]
        if A_REWARD == 7:
            return A_PRIME_POS_LEFT, 0.7, 7
        else:
            return A_PRIME_POS_RIGHT, 0.3, 13

    next_state = (np.array(state) + action).tolist()
    # 转成列表

    x, y = next_state

    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0

    return next_state, 1, reward


def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    for (i, j), val in np.ndenumerate(image):
        if [i, j] == A_POS:
            val = str(val) + "(A)"
        if [i, j] == A_PRIME_POS_LEFT:
            val = str(val) + "(A’_left)"
        if [i, j] == A_PRIME_POS_RIGHT:
            val = str(val) + "(A‘_right)"
        if [i, j] == B_POS:
            val = str(val) + "(B)"
        if [i, j] == B_PRIME_POS:
            val = str(val) + "(B')"

        tb.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i + 1, loc='right', edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center', edgecolor='none', facecolor='none')

    ax.add_table(tb)

def draw_policy(optimal_values):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = optimal_values.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(optimal_values):
        next_vals = []
        for action in ACTIONS:
            next_state, _, _ = step([i, j], action)
            next_vals.append(optimal_values[next_state[0], next_state[1]])

        best_actions = np.where(next_vals == np.max(next_vals))[0]
        val = ''
        for ba in best_actions:
            val += ACTIONS_FIGS[ba]

        # add state labels
        if [i, j] == A_POS:
            val = str(val) + " (A)"
        if [i, j] == A_PRIME_POS_LEFT:
            val = str(val) + "(A'_left)"
        if [i, j] == A_PRIME_POS_RIGHT:
            val = str(val) + "(A'_right)"
        if [i, j] == B_POS:
            val = str(val) + " (B)"
        if [i, j] == B_PRIME_POS:
            val = str(val) + " (B')"

        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

    # Row and column labels...
    for i in range(len(optimal_values)):
        tb.add_cell(i, -1, width, height, text=i + 1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center',
                    edgecolor='none', facecolor='none')

    ax.add_table(tb)

def figure_3_2():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                for action in ACTIONS:
                    (next_i, next_j), action_pro, reward = step([i, j], action)
                    # bellman equation
                    new_value[i, j] += ACTION_PROB * action_pro * (reward + DISCOUNT * value[next_i, next_j])
        if np.sum(np.abs(value - new_value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('../figure_3_2_new.png')
            plt.close()
            break
        value = new_value

def figure_3_2_linear_system():
    '''
    Here we solve the linear system of equations to find the exact solution.
    We do this by filling the coefficients for each of the states with their respective right side constant.
    '''
    A = -1 * np.eye(WORLD_SIZE * WORLD_SIZE)
    b = np.zeros(WORLD_SIZE * WORLD_SIZE)
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            s = [i, j]  # current state
            index_s = np.ravel_multi_index(s, (WORLD_SIZE, WORLD_SIZE))
            for a in ACTIONS:
                s_, ac_, r = step(s, a)
                index_s_ = np.ravel_multi_index(s_, (WORLD_SIZE, WORLD_SIZE))

                A[index_s, index_s_] += ACTION_PROB * ac_ * DISCOUNT
                b[index_s] -= ACTION_PROB * ac_ * r

    x = np.linalg.solve(A, b)
    draw_image(np.round(x.reshape(WORLD_SIZE, WORLD_SIZE), decimals=2))
    plt.savefig('../figure_3_2_linear_system_new.png')
    plt.close()

def figure_3_5():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                values = []
                for action in ACTIONS:
                    (next_i, next_j), action_pro, reward = step([i, j], action)
                    # value iteration
                    values.append(action_pro * (reward + DISCOUNT * value[next_i, next_j]))
                new_value[i, j] = np.max(values)
        if np.sum(np.abs(new_value - value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('../figure_3_5_new.png')
            plt.close()
            draw_policy(new_value)
            plt.savefig('../figure_3_5_policy_new.png')
            plt.close()
            break
        value = new_value

if __name__ == '__main__':
    figure_3_2()
    figure_3_2_linear_system()
    figure_3_5()