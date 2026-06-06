"""Environment wrappers for the regular-vs-continuous comparison.

Both conditions run on the *same* deterministic 4x4 FrozenLake grid (gym's default 4x4
map is byte-for-byte the ``TILES`` map used by ``ContinuousFrozenLake``), with the *same*
reward schedule applied by tile type. The only difference is episode handling:

  - ``EpisodicFrozenLake``  : goal/hole are terminal -> done=True -> caller resets.
  - ``ContinuousFrozenLake``: no terminals, no resets (the class under test).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import gymnasium as gym

# ContinuousFrozenLake calls env.render() every step to keep an optional renderer in sync;
# with no render_mode that emits a one-time gym warning we don't care about here.
warnings.filterwarnings("ignore", message=".*render method without specifying.*")

# Make the project's ``src/`` importable so we exercise the *real* ContinuousFrozenLake.
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuous_frozen_lake import TILES, ContinuousFrozenLake  # noqa: E402

ENV_NAME = "FrozenLake-v1"
GOAL_STATE = TILES.index("G")  # 15

# Optimal deterministic path S -> G on this map (6 steps): Down,Down,Right,Down,Right,Right
OPTIMAL_PATH = [0, 4, 8, 9, 13, 14, 15]

RewardSchedule = tuple[float, float, float]  # (goal_r, hole_r, step_r)
DEFAULT_REWARDS: RewardSchedule = (10.0, -10.0, -1.0)


def make_continuous(reward_schedule: RewardSchedule, seed: int) -> ContinuousFrozenLake:
    """Build the continuous env. A gym env is required because ContinuousFrozenLake reads
    its observation/action spaces from it (and would use it for rendering if enabled)."""
    genv = gym.make(ENV_NAME, is_slippery=False)
    genv.reset(seed=seed)
    return ContinuousFrozenLake(reward_schedule=reward_schedule, env=genv)


class EpisodicFrozenLake:
    """Regular gym FrozenLake-v1 with rewards remapped to match the continuous schedule.

    Gym FrozenLake natively gives reward 1 at the goal and 0 everywhere else (including
    holes), and terminates on both goal and hole. We recover goal-vs-hole from the tile
    so both conditions see an identical reward signal.
    """

    def __init__(self, reward_schedule: RewardSchedule, seed: int | None = None) -> None:
        self.goal_r, self.hole_r, self.step_r = reward_schedule
        self.env = gym.make(ENV_NAME, is_slippery=False)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._seed = seed

    def reset(self) -> tuple[int, None]:
        obs, _ = self.env.reset(seed=self._seed)
        self._seed = None  # only seed the very first reset
        return int(obs), None

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        obs, _gym_reward, terminated, truncated, info = self.env.step(action)
        tile = TILES[obs]
        if tile == "G":
            reward = self.goal_r
        elif tile == "H":
            reward = self.hole_r
        else:
            reward = self.step_r
        done = bool(terminated or truncated)
        return int(obs), reward, done, info

    def close(self) -> None:
        self.env.close()
