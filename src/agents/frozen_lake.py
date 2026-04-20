"""HTM Brain consumer for Gymnasium FrozenLake-v1 environment.

Wires a Brain with:
  - CategoryEncoder InputField for discrete state observations (16 tiles)
  - ColumnField for temporal memory / sequence learning
  - ValueFieldMixin go/nogo fields for TD-based reward signaling
  - OutputField for stochastic action selection (4 actions)

Uses ContinuousFrozenLake — no episode boundaries, the agent moves
continuously on the grid and receives rewards for holes/goals/steps.

Usage:
    python src/frozen_lake.py
    python src/frozen_lake.py --viz
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
from continuous_frozen_lake import ContinuousFrozenLake


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
NUM_STEPS = 50_000
PRINT_EVERY = 100


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


def main(viz: bool = False) -> None:
    render_env = gym.make(ENV_NAME, is_slippery=False, render_mode="human",
                          reward_schedule=(10, -10, -1))
    env = ContinuousFrozenLake(
        reward_schedule=(10, -10, -1),
        env=render_env,
    )
    agent = build_agent()

    if viz:
        from visualizer.app import HTMVisualizer
        HTMVisualizer(agent.brain, title="FrozenLake HTM").start()

    obs = env.reset()
    last_reward = 0.0
    total_reward = 0.0
    recent_reward = 0.0

    for step in range(1, NUM_STEPS + 1):
        action = agent.step(obs, reward=last_reward, learn=True)
        obs, reward, was_terminal = env.step(action)
        last_reward = float(reward)
        total_reward += last_reward
        recent_reward += last_reward

        if step % PRINT_EVERY == 0:
            print(
                f"Step {step:>6d} | "
                f"Recent avg reward: {recent_reward / PRINT_EVERY:+.2f} | "
                f"Total reward: {total_reward:+.1f}"
            )
            recent_reward = 0.0

    env.close()


if __name__ == "__main__":
    main(viz=True or "--viz" in sys.argv)
