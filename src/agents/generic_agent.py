from __future__ import annotations

from collections import defaultdict
import collections
from pathlib import Path
import random
import sys
from typing import cast

import gymnasium as gym
from gymnasium import spaces
from tabulate import tabulate

from agents.state_brain_builder import state_to_inputs
from continuous_env import ContinuousEnv
from core.brain import Brain
from core.HTM import ColumnField, InputField
from encoder_layer.category import CategoryParametersNew

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

value_map = defaultdict(lambda: 0)  # Maps state to value estimate
action_value_map = defaultdict(lambda: 0)  # Maps (state, action) to value estimate

def choose_action(state, action_space, epsilon=0.0):
    if random.random() < epsilon:
        return action_space.sample()

    best_action = max(range(action_space.n), key=lambda a: action_value_map[(state, a)])
    return best_action

def td_error(reward, current_state, action, next_state, next_action, gamma=0.95):
    return reward + gamma * action_value_map[(next_state, next_action)] - action_value_map[(current_state, action)]

def update_value(current_state, action, error, alpha=0.1):
    action_value_map[(current_state, action)] += alpha * error

def window_average_plateaued(values, window_size: int, abs_tolerance: float, rel_tolerance: float):
    if len(values) < 2 * window_size:
        return False, None, None

    samples = list(values)
    previous_window = samples[-2 * window_size:-window_size]
    recent_window = samples[-window_size:]

    previous_avg = sum(previous_window) / window_size
    recent_avg = sum(recent_window) / window_size
    tolerance = max(abs_tolerance, rel_tolerance * max(abs(previous_avg), abs(recent_avg), 1.0))
    return abs(recent_avg - previous_avg) <= tolerance, previous_avg, recent_avg

def reward_average_plateaued(reward_history, window_size: int, abs_tolerance: float, rel_tolerance: float):
    return window_average_plateaued(reward_history, window_size, abs_tolerance, rel_tolerance)

def td_error_average_plateaued(td_error_history, window_size: int, abs_tolerance: float, rel_tolerance: float):
    absolute_errors = [abs(error) for error in td_error_history]
    return window_average_plateaued(absolute_errors, window_size, abs_tolerance, rel_tolerance)

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

def build_brain():
    encoder_params = CategoryParametersNew(size=100, category_list=list(range(16)))
    state = InputField(encoder_params=encoder_params)
    for s in range(16):
        state.encoder.encode(s)
    column = ColumnField(input_fields=[state], 
                         cells_per_column=8,
                         non_spatial=True)
    return Brain(fields={"state": state, "column": column})

def main(
    viz: bool = False,
    steps: int = 1000,
    reward_window_size: int = 50,
    td_error_window_size: int = 50,
    plateau_patience: int = 5,
    abs_tolerance: float = 0.05,
    rel_tolerance: float = 0.01,
    td_abs_tolerance: float = 0.05,
    td_rel_tolerance: float = 0.01,
) -> None:
    genv = gym.make("FrozenLake-v1", is_slippery=False)
    env = ContinuousEnv(env=genv)
    state, _ = env.reset()
    # brain = build_brain(observation_space=env.observation_space, state_sample=state)
    from agents.state_brain_builder import build_typed_brain    
    brain = build_typed_brain(observation_space=env.observation_space, state_sample=state)
    action_space = cast(spaces.Discrete, env.action_space)

    if viz:
        from visualizer.app import HTMVisualizer

        HTMVisualizer(brain).start()

    brain.step(state_to_inputs(state))
    td_state = binary_vector_to_number(brain["column"].state("active"))

    action = choose_action(td_state, action_space)
    reward_history = collections.deque(maxlen=2 * reward_window_size)
    td_error_history = collections.deque(maxlen=2 * td_error_window_size)
    plateau_streak = 0
    for i in range(steps):
        next_state, reward, done, _ = env.step(action)
        reward_history.append(float(reward))
        brain.step(state_to_inputs(next_state))
        td_next_state = binary_vector_to_number(brain["column"].state("active"))

        next_action = choose_action(td_next_state, action_space)

        error = td_error(reward, td_state, action, td_next_state, next_action)
        td_error_history.append(float(error))
        update_value(td_state, action, error)
        print_action_value_table(action_value_map, action_space.n)
        print(f"State: {td_next_state} | Action: {next_action} | Reward: {reward:.2f} | TD Error: {error:.2f}")
        # if not viz:
        #     input()

        if i + 1 >= 2 * max(reward_window_size, td_error_window_size):
            reward_plateaued, reward_previous_avg, reward_recent_avg = reward_average_plateaued(
                reward_history,
                reward_window_size,
                abs_tolerance,
                rel_tolerance,
            )
            td_plateaued, td_previous_avg, td_recent_avg = td_error_average_plateaued(
                td_error_history,
                td_error_window_size,
                td_abs_tolerance,
                td_rel_tolerance,
            )

            if reward_plateaued and td_plateaued:
                plateau_streak += 1
                if plateau_streak >= plateau_patience:
                    print(
                        "Ending at step "
                        f"{i} after reward plateau and TD-error plateau: "
                        f"reward previous avg={reward_previous_avg:+.3f}, "
                        f"reward recent avg={reward_recent_avg:+.3f}, "
                        f"td previous avg={td_previous_avg:+.3f}, "
                        f"td recent avg={td_recent_avg:+.3f}"
                    )
                    break
            else:
                plateau_streak = 0

        td_state = td_next_state
        action = next_action


if __name__ == "__main__":
    main()