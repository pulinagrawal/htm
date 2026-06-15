"""Tabular (no-HTM) ablation of the TD agent on Continuous FrozenLake.

This mirrors ``td_agent_htm.py`` exactly — same reward schedule, gamma, alpha,
plateau rule, and returned metrics dict — but removes HTM entirely: the TD-state
key is the raw discrete FrozenLake index (0-15) instead of a ColumnField SDR.

It exists so HTM vs. tabular runs are directly A/B comparable (the older
``td_agent.py`` uses a different plateau threshold and does not track cumulative
reward, so it is not comparable to the HTM results).
"""

import collections
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
from tabulate import tabulate
from tqdm import tqdm

# Make ``src`` importable so this runs standalone regardless of cwd.
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuous_frozen_lake import ContinuousFrozenLake  # noqa: E402

import random  # noqa: E402

value_map = defaultdict(lambda: 0)  # Maps state to value estimate
action_value_map = defaultdict(lambda: 0)  # Maps (state, action) to value estimate


def initialize_state_actions(state, action_space):
    """Ensure every action for a state exists in action_value_map with a zero value."""
    for action in range(action_space.n):
        action_value_map.setdefault((state, action), 0)


def choose_action(state, action_space, epsilon=0.0):
    """Greedy (epsilon-greedy) action selection over the raw tabular state.

    No episodic-memory generalization: raw int states are exact keys, so there
    is nothing to average over neighbours.
    """
    initialize_state_actions(state, action_space)
    if random.random() < epsilon:
        return action_space.sample()
    return max(range(action_space.n), key=lambda a: action_value_map[(state, a)])


def td_error(reward, previous_state, action, current_state, action_taken, gamma=0.95):
    return reward + gamma * action_value_map[(current_state, action_taken)] - action_value_map[(previous_state, action)]


def update_value(previous_state, action_taken, td_error, alpha=0.1):
    action_value_map[(previous_state, action_taken)] += alpha * td_error


def print_action_value_table(action_value_map, action_space_n):
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


def measure_run(episodic_memory=False, max_steps=1000, seed=None, verbose=True):
    """Run one tabular SARSA episode-free trajectory.

    ``episodic_memory`` is accepted for call-compatibility with the HTM harness
    but is ignored — the tabular agent has no SDR neighbourhood to generalize
    over.
    """
    global value_map, action_value_map
    value_map = defaultdict(lambda: 0)
    action_value_map = defaultdict(lambda: 0)
    if seed is not None:
        random.seed(seed)

    genv = gym.make('FrozenLake-v1', is_slippery=False)
    env = ContinuousFrozenLake(env=genv)
    if seed is not None:
        env.action_space.seed(seed)

    reset_result = env.reset()
    state = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    td_previous_state = state
    action = choose_action(td_previous_state, env.action_space)

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

        td_current_state = current_state
        next_action = choose_action(td_current_state, env.action_space)

        error = td_error(reward, td_previous_state, action, td_current_state, next_action)
        update_value(td_previous_state, action, error)

        last_states.append(current_state)
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

    env.close()
    return {
        'steps_taken': steps_taken,
        'plateau_detected': plateau_detected,
        'rewards': rewards,
        'cumulative_rewards': cumulative_rewards,
    }


def measure_steps(episodic_memory=False, max_steps=1000, seed=None, verbose=True):
    return measure_run(
        episodic_memory=episodic_memory,
        max_steps=max_steps,
        seed=seed,
        verbose=verbose,
    )['steps_taken']


if __name__ == "__main__":
    import numpy

    steps = []
    nsteps = 100
    for run in tqdm(range(nsteps)):
        steps.append(measure_steps(seed=run, verbose=False))
    print(f"Average steps to plateau over {nsteps} runs: {numpy.mean(steps):.2f}")
    print(f"Standard deviation of steps to plateau: {numpy.std(steps):.2f}")
