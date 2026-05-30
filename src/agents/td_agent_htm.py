import collections

from tqdm import tqdm

from agents.state import State
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

def choose_action(state, action_space, epsilon=0.0, episodic_memory=True):
    """ for states that are similar as defined by distance we want to choose action that has the highest value estimate for any of those states
    """
    initialize_state_actions(state, action_space)

    if random.random() < epsilon:
        return action_space.sample()
    if episodic_memory:
        nearby_state_actions = list(
            filter(lambda s: s[0].distance(state) < 0.04, action_value_map.keys())
        )
        action_scores = []
        for action in range(action_space.n):
            action_keys = [
                key for key in nearby_state_actions if key[1] == action
            ]
            if not action_keys:
                continue
            average_value = sum(action_value_map[key] for key in action_keys) / len(action_keys)
            action_scores.append((action, average_value))

        if not action_scores:
            return action_space.sample()

        best_score = max(score for _, score in action_scores)
        best_actions = [action for action, score in action_scores if score == best_score]
        return random.choice(best_actions)
    else:
        return max(range(action_space.n), key=lambda a: action_value_map[(state, a)])

def td_error(reward, previous_state, action, current_state, action_taken, gamma=0.95):
    return reward + gamma * action_value_map[(current_state, action_taken)] - action_value_map[(previous_state, action)]

def update_value(previous_state, action_taken, td_error, alpha=0.1):
    action_value_map[(previous_state, action_taken)] += alpha * td_error

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
  
def build_brain():
    state_encoder_params = CategoryParametersNew(
        size=100,
        active_bits_per_category=5,
        category_list=list(range(NUM_STATES)),
    )
    state_field = InputField(encoder_params=state_encoder_params)
    action_encoder_params = CategoryParametersNew(
        size=50,
        active_bits_per_category=5,
        category_list=list(range(5)),
    )
    action_field = InputField(encoder_params=action_encoder_params)

    for s in range(5):
        action_field.encoder.encode(s)

    column_field = ColumnField(
        input_fields=[state_field],
        non_spatial=True,
        cells_per_column=4
    )
    # column_field1 = ColumnField(
    #     input_fields=[column_field],
    #     non_spatial=False,
    #     active_boost=True,
    #     cells_per_column=4
    # )
    return Brain({'state': state_field,
                  'action':action_field,
                  'column': column_field,
                #   'column1': column_field1
                 })

def binary_vector_to_number(vec):
    """Convert a binary vector (list of 0s and 1s) to an integer."""
    if isinstance(vec, dict):
        return State(vec)
    return '|'.join(str(i) for i, bit in enumerate(vec) if bit)

def measure_run(episodic_memory=True, max_steps=1000, seed=None, verbose=False):
    global value_map, action_value_map
    value_map = defaultdict(lambda: 0)  # Maps state to value estimate
    action_value_map = defaultdict(lambda: 0)  # Maps (state, action) to value estimate
    if seed is not None:
        random.seed(seed)
    viz = False
    genv = gym.make('FrozenLake-v1', is_slippery=False)#, render_mode='human')
    env = ContinuousFrozenLake(env=genv)
    if seed is not None:
        env.action_space.seed(seed)
    brain = build_brain()
    reset_result = env.reset()
    state = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    if viz:
        from visualizer.app import HTMVisualizer
        HTMVisualizer(brain).start()

    brain.step({'state': state})

    def get_state():
        return {'active': brain['column'].state('active')}

    td_previous_state = binary_vector_to_number(get_state())

    action = choose_action(td_previous_state, env.action_space, episodic_memory=episodic_memory)
    last_states = collections.deque(maxlen=100)
    rewards = []
    cumulative_rewards = []
    reward_total = 0.0
    plateau_detected = False

    steps_taken = 0
    for i in range(1, max_steps + 1):
        steps_taken = i
        step_result = env.step(action)
        current_state = step_result[0]
        reward = step_result[1]
        rewards.append(reward)
        reward_total += reward
        cumulative_rewards.append(reward_total)
        td_current_state = binary_vector_to_number(get_state())
        next_action = choose_action(
            td_current_state,
            env.action_space,
            episodic_memory=episodic_memory,
        )

        brain.step({'state': current_state, 'action': next_action})
        error = td_error(reward, td_previous_state, action, td_current_state, next_action)
        
        update_value(td_previous_state, action, error)
        # print_action_value_table(action_value_map, env.env.action_space.n)
        # print(f"State: {td_next_state} | Action: {next_action} | Reward: {reward:.2f} | TD Error: {error:.2f}")

        last_states.append(current_state)
        # check if majority of last states are the same to detect plateau
        if len(last_states) == last_states.maxlen:
            most_common_state, count = collections.Counter(last_states).most_common(1)[0]
            if count >= 0.5 * len(last_states) and most_common_state == 15:
                plateau_detected = True
                if verbose:
                    print(f"Plateau detected at state {most_common_state} with count {count}/{last_states.maxlen}")
                    print("Last states leading to plateau:", list(last_states))
                    print('ending step number:', i)
                break
        state = current_state
        td_previous_state = td_current_state
        action = next_action
        # if not viz:
        #     input()
    env.close()
    return {
        'steps_taken': steps_taken,
        'plateau_detected': plateau_detected,
        'rewards': rewards,
        'cumulative_rewards': cumulative_rewards,
    }


def measure_steps(episodic_memory=True, max_steps=1000, seed=None, verbose=False):
    return measure_run(
        episodic_memory=episodic_memory,
        max_steps=max_steps,
        seed=seed,
        verbose=verbose,
    )['steps_taken']

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
