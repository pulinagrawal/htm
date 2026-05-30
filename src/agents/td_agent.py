import collections

from tqdm import tqdm

from agents.state import State
from continuous_env import ContinuousEnv
import gymnasium as gym
from collections import defaultdict
from tabulate import tabulate
import random

from continuous_frozen_lake import ContinuousFrozenLake
from encoder_layer.category import CategoryParametersNew
from core.HTM import ColumnField, InputField
from core.brain import Brain

value_map = defaultdict(lambda: 0)  # Maps state to value estimate
action_value_map = defaultdict(lambda: 0)  # Maps (state, action) to value estimate

NUM_STATES = 16 


def initialize_state_actions(state, action_space):
    """Ensure every action for a state exists in action_value_map with a zero value."""
    for action in range(action_space.n):
        action_value_map.setdefault((state, action), 0)

def choose_action(state, action_space, epsilon=0.0):
    """ for states that are similar as defined by distance we want to choose action that has the highest value estimate for any of those states
    """
    initialize_state_actions(state, action_space)

    if random.random() < epsilon:
        return action_space.sample()

    best_action = max(range(action_space.n), key=lambda a: action_value_map[(state, a)])
    return best_action

def td_error(reward, current_state, action, next_state, next_action, gamma=0.95):
    return reward + gamma * action_value_map[(next_state, next_action)] - action_value_map[(current_state, action)]

def update_value(current_state, action, error, alpha=0.1):
    action_value_map[(current_state, action)] += alpha * error

def print_action_value_table(action_value_map, action_space_n):
    # Collect all unique states
    states = sorted(set(state for (state, _) in action_value_map.keys()))
    actions = list(range(action_space_n))

    table = []
    for state in states:
        row = [state]
        for action in actions:
            value = action_value_map.get((state, action), 0)
            row.append(f"{value:.2f}")
        table.append(row)
    headers = ["State"] + [f"Action {a}" for a in actions]
    print(tabulate(table, headers=headers, tablefmt="pretty"))
  

def binary_vector_to_number(vec):
    """Convert a binary vector (list of 0s and 1s) to an integer."""
    return '|'.join(str(i) for i, bit in enumerate(vec) if bit)

def measure_steps():
    global value_map, action_value_map
    value_map = defaultdict(lambda: 0)  # Maps state to value estimate
    action_value_map = defaultdict(lambda: 0)  # Maps (state, action) to value estimate
    
    genv = gym.make('FrozenLake-v1', is_slippery=False)#, render_mode='human')
    env = ContinuousFrozenLake(env=genv)

    state, _ = env.reset()
    action = choose_action(state, env.env.action_space)

    last_states = collections.deque(maxlen=100)

    for i in range(1000):
        next_state, reward, done, _ = env.step(action)
        next_action = choose_action(next_state, env.env.action_space)

        error = td_error(reward, state, action, next_state, next_action)
        update_value(state, action, error)

        print_action_value_table(action_value_map, env.env.action_space.n)
        print(f"State: {next_state} | Action: {next_action} | Reward: {reward:.2f} | TD Error: {error:.2f}")

        last_states.append(next_state)
        # check if majority of last states are the same to detect plateau
        if len(last_states) == last_states.maxlen:
            most_common_state, count = collections.Counter(last_states).most_common(1)[0]
            if count > 0.5 * last_states.maxlen and most_common_state == 15:
                print(f"Plateau detected at state {most_common_state} with count {count}/{last_states.maxlen}")
                print("Last states leading to plateau:", list(last_states))
                print('ending step number:', i )
                break

        state = next_state
        action = next_action

    return i

if __name__ == "__main__":
    steps = []
    import numpy
    nsteps = 100
    for run in tqdm(range(nsteps)):
        steps_to_plateau = measure_steps()
        steps.append(steps_to_plateau)
    # Avg and Std
    print(f"Average steps to plateau over {nsteps} runs: {numpy.mean(steps):.2f}")
    print(f"Standard deviation of steps to plateau: {numpy.std(steps):.2f}")
