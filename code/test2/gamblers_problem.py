import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

# goal
GOAL = 200

# all states, including state 0 and state 200
STATES = np.arange(GOAL + 1)

# probability of head
HEAD_PROB = 0.6

def figure_All_in():
    state_value = np.zeros(GOAL + 1)
    state_value[GOAL] = 1.0

    sweeps_history = []

    while True:
        old_state_value = state_value.copy()
        sweeps_history.append(old_state_value)


        for state in STATES[1:GOAL]:
            if (state <= GOAL/2):
                action = state
            else:
                action = (GOAL-state)

            new_value = HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action]
            state_value[state] = new_value

        delta = abs(state_value - old_state_value).max()
        if delta < 1e-9:
            sweeps_history.append(state_value)
            break

    plt.figure(figsize=(10, 10))

    for sweep, state_value in enumerate(sweeps_history):
        plt.plot(state_value, label='sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')

    plt.savefig('../figure_All_in.png')
    plt.close()

def figure_one_dollar():
    state_value = np.zeros(GOAL + 1)
    state_value[GOAL] = 1.0

    sweeps_history = []

    while True:
        old_state_value = state_value.copy()
        sweeps_history.append(old_state_value)


        for state in STATES[1:GOAL]:
            action = 1
            new_value = HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action]
            state_value[state] = new_value

        delta = abs(state_value - old_state_value).max()
        if delta < 1e-9:
            sweeps_history.append(state_value)
            break

    plt.figure(figsize=(10, 10))

    for sweep, state_value in enumerate(sweeps_history):
        plt.plot(state_value, label='sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')

    plt.savefig('../figure_one_dollar.png')
    plt.close()

def figure_Two_dollar():
    state_value = np.zeros(GOAL + 1)
    state_value[GOAL] = 1.0

    sweeps_history = []

    while True:
        old_state_value = state_value.copy()
        sweeps_history.append(old_state_value)


        for state in STATES[1:GOAL]:
            if state == 1 or state == GOAL - 1:
                action = 1
            else:
                action = 2
            new_value = HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action]
            state_value[state] = new_value

        delta = abs(state_value - old_state_value).max()
        if delta < 1e-9:
            sweeps_history.append(state_value)
            break

    plt.figure(figsize=(10, 10))

    for sweep, state_value in enumerate(sweeps_history):
        plt.plot(state_value, label='sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')

    plt.savefig('../figure_Two_dollar.png')
    plt.close()

if __name__ == '__main__':
    figure_All_in()
    figure_one_dollar()
    figure_Two_dollar()