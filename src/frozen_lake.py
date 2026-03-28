"""HTM Brain consumer for Gymnasium FrozenLake-v1 environment.

Wires a Brain with:
  - CategoryEncoder InputField for discrete state observations (16 tiles)
  - ColumnField for temporal memory / sequence learning
  - ValueFieldMixin go/nogo fields for TD-based reward signaling
  - OutputField for stochastic action selection (4 actions)

Usage:
    python src/frozen_lake.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from typing import Any

import gymnasium as gym

from core.brain import Brain
from core.HTM import ColumnField, InputField, OutputField
from core.sungur import ValueField
from encoder_layer.category import CategoryParametersNew
from gym_adapter import GymBrain


# -- Environment ----------------------------------------------------------------

ENV_NAME = "FrozenLake-v1"
NUM_STATES = 16
NUM_ACTIONS = 4
ACTION_NAMES = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

# -- Encoder parameters ---------------------------------------------------------

STATE_ENCODER_SIZE = 256
STATE_ACTIVE_BITS = 10

ACTION_ENCODER_SIZE = 128
ACTION_ACTIVE_BITS = 10

# -- HTM parameters -------------------------------------------------------------

CELLS_PER_COLUMN = 8
NUM_EPISODES = 5_000
MAX_STEPS_PER_EPISODE = 100
PRINT_EVERY = 5


def build_brain() -> Brain:
    """Construct the HTM brain for FrozenLake."""

    # State encoder: one unique SDR per tile (0-15)
    state_encoder_params = CategoryParametersNew(
        size=STATE_ENCODER_SIZE,
        active_bits_per_category=STATE_ACTIVE_BITS,
        category_list=list(range(NUM_STATES)),
    )
    state_field = InputField(encoder_params=state_encoder_params)

    # Pre-register all 16 states so the encoder cache is warm
    for s in range(NUM_STATES):
        state_field.encoder.encode(s)

    # Temporal memory layer
    column_field = ColumnField(
        input_fields=[state_field],
        non_spatial=True,
        cells_per_column=CELLS_PER_COLUMN,
    )

    # Go / NoGo value fields for TD reward signaling
    go_field = ValueField(input_fields=[column_field], non_spatial=True, cells_per_column=CELLS_PER_COLUMN)
    nogo_field = ValueField(input_fields=[column_field], non_spatial=True, cells_per_column=CELLS_PER_COLUMN)

    column_field.go_field = go_field
    column_field.nogo_field = nogo_field

    # Action output field driven by column layer activity
    action_encoder_params = CategoryParametersNew(
        size=ACTION_ENCODER_SIZE,
        active_bits_per_category=ACTION_ACTIVE_BITS,
        category_list=list(range(NUM_ACTIONS)),
    )
    action_field = OutputField(
        input_field=column_field,
        encoder_params=action_encoder_params,
        size=ACTION_ENCODER_SIZE,
    )

    # Pre-register action encodings (0-3) so decode can map back
    for a in range(NUM_ACTIONS):
        action_field.encoder.encode(a)

    return Brain({  # type: ignore[arg-type]  # ValueFieldMixin isn't a Field subclass
        "state": state_field,
        "columns": column_field,
        "go": go_field,
        "nogo": nogo_field,
        "action": action_field,
    })


def pick_action(brain: Brain) -> int:
    """Decode the OutputField activation into a discrete action (legacy helper)."""
    action_field: OutputField = brain.fields["action"]  # type: ignore[assignment]
    result = action_field.decode()
    value = result["value"]
    if value is not None and 0 <= int(value) < NUM_ACTIONS:
        return int(value)
    import random
    return random.randint(0, NUM_ACTIONS - 1)


# -- FrozenLake adapters -------------------------------------------------------

def obs_to_inputs(obs: Any) -> dict[str, Any]:
    """Convert FrozenLake observation (int 0-15) to Brain input dict."""
    return {"state": float(obs)}


def behavior_to_action(behavior: dict[str, Any]) -> int:
    """Convert Brain behavior dict to a discrete FrozenLake action."""
    value = behavior.get("action")
    if value is not None and 0 <= int(value) < NUM_ACTIONS:
        return int(value)
    import random
    return random.randint(0, NUM_ACTIONS - 1)


def build_agent() -> GymBrain:
    """Construct the HTM brain wrapped for Gymnasium use."""
    brain = build_brain()
    return GymBrain(brain, obs_to_inputs, behavior_to_action)


def run_episode(agent: GymBrain, env: gym.Env, learn: bool = True) -> tuple[float, int]:
    """Run a single FrozenLake episode."""
    obs, _ = env.reset()
    total_reward = 0.0

    for step in range(MAX_STEPS_PER_EPISODE):
        action = agent.step(obs, reward=total_reward, learn=learn)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            agent.step(obs, reward=total_reward, learn=learn)
            print(f"Episode ended after {step + 1} steps with reward {total_reward:.2f}")
            return total_reward, step + 1

    return total_reward, MAX_STEPS_PER_EPISODE


def main(viz: bool = False) -> None:
    env = gym.make(ENV_NAME, is_slippery=False, render_mode="human",
                   reward_schedule=(100, -30, -1))
    agent = build_agent()

    if viz:
        from visualizer.app import HTMVisualizer
        HTMVisualizer(agent.brain, title="FrozenLake HTM").start()

    wins = 0
    recent_wins = 0
    recent_window = PRINT_EVERY

    for episode in range(1, NUM_EPISODES + 1):
        reward, steps = run_episode(agent, env, learn=True)
        if reward > 0:
            wins += 1
            recent_wins += 1

        if episode % recent_window == 0:
            win_rate = recent_wins / recent_window
            print(
                f"Episode {episode:>5d} | "
                f"Recent win rate: {win_rate:.2%} | "
                f"Total wins: {wins}/{episode}"
            )
            recent_wins = 0

    # Evaluation (no learning)
    eval_episodes = 100
    eval_wins = 0
    for _ in range(eval_episodes):
        reward, _ = run_episode(agent, env, learn=False)
        if reward > 0:
            eval_wins += 1

    print(f"\nEvaluation ({eval_episodes} episodes, no learning): "
          f"Win rate = {eval_wins / eval_episodes:.2%}")

    env.close()


if __name__ == "__main__":
    main(viz=True or "--viz" in sys.argv)
