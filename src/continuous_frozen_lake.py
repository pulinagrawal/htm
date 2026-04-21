"""Continuous FrozenLake — grid movement with no episode boundaries.

Implements the 4x4 FrozenLake grid directly, bypassing gymnasium's
terminal-state absorption. Uses an optional gym env purely for rendering.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym

GRID_ROWS = 4
GRID_COLS = 4

TILES = [
    "S", "F", "F", "F",
    "F", "H", "F", "H",
    "F", "F", "F", "H",
    "H", "F", "F", "G",
]

LABELS = [t + " " for t in TILES]


class ContinuousFrozenLake:
    """FrozenLake as a continuous grid — no terminal states, no episodes.

    Handles movement and rewards directly. Uses an optional gym env
    purely for rendering (kept in sync via unwrapped.s + render()).
    """

    def __init__(
        self,
        reward_schedule: tuple[float, float, float] = (10, -10, -1),
        env: Optional[gym.Env] = None,
    ) -> None:
        self.goal_r, self.hole_r, self.step_r = reward_schedule
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env = env
        self.s = 0

    def reset(self) -> int:
        self.s = 0
        if self.env:
            self.env.reset()
        return self.s, None

    def step(self, action: int) -> tuple[int, float, bool]:
        """Move on the grid, return (new_state, reward, was_terminal).

        was_terminal indicates the agent landed on H or G (for logging),
        but the grid keeps going regardless.
        """
        row, col = divmod(self.s, GRID_COLS)
        if action == 0:
            col = max(col - 1, 0)            # Left
        elif action == 1:
            row = min(row + 1, GRID_ROWS - 1)  # Down
        elif action == 2:
            col = min(col + 1, GRID_COLS - 1)  # Right
        elif action == 3:
            row = max(row - 1, 0)            # Up

        self.s = row * GRID_COLS + col
        tile = TILES[self.s]

        if tile == "G":
            reward = self.goal_r
        elif tile == "H":
            reward = self.hole_r
        else:
            reward = self.step_r

        was_terminal = tile in ("G", "H")

        # Sync renderer
        if self.env:
            try:
                self.env.unwrapped.s = self.s
                self.env.unwrapped.lastaction = action
                self.env.render()
            except Exception:
                self.env.reset()
                self.env.unwrapped.s = self.s

        return self.s, reward, was_terminal, None

    def close(self) -> None:
        if self.env:
            self.env.close()
