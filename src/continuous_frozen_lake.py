"""Continuous FrozenLake — grid movement with no episode boundaries.

Implements the 4x4 FrozenLake grid directly, bypassing gymnasium's
terminal-state absorption. Uses an optional gym env purely for rendering.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

# Keep module-level defaults for backwards compatibility and external imports
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
        grid_rows: Optional[int] = None,
        grid_cols: Optional[int] = None,
        tiles: Optional[list[str]] = None,
        labels: Optional[list[str]] = None,
    ) -> None:
        self.goal_r, self.hole_r, self.step_r = reward_schedule
        self.env = env

        # Determine grid shape and tile labels. Priority:
        # 1. explicit `tiles` + `grid_rows/grid_cols` args
        # 2. read from `env.unwrapped.desc` (gym FrozenLake-style)
        # 3. fall back to module-level defaults
        self.tiles = None
        self.labels = None

        if tiles is not None:
            self.tiles = list(tiles)

        if self.env is not None:
            try:
                unwrapped = self.env.unwrapped
                desc = getattr(unwrapped, "desc", None)
                if desc is not None:
                    arr = np.array(desc)
                    # desc can be bytes; normalize to str
                    flat = []
                    for c in arr.flatten():
                        if isinstance(c, (bytes, bytearray)):
                            flat.append(c.decode())
                        else:
                            flat.append(str(c))
                    # only set tiles if not explicitly provided
                    if self.tiles is None:
                        self.tiles = flat
                    # set shape if not provided
                    if grid_rows is None or grid_cols is None:
                        grid_rows = arr.shape[0]
                        grid_cols = arr.shape[1]
            except Exception:
                # ignore and fall back to explicit args or defaults
                pass

        # Finalize shape and labels
        self.grid_rows = grid_rows if grid_rows is not None else GRID_ROWS
        self.grid_cols = grid_cols if grid_cols is not None else GRID_COLS

        if self.tiles is None:
            self.tiles = list(TILES)

        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = [t + " " for t in self.tiles]

        # Expose observation/action spaces when env present for compatibility
        if self.env is not None:
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
        else:
            self.observation_space = None
            self.action_space = None

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
        row, col = divmod(self.s, self.grid_cols)
        if action == 0:
            col = max(col - 1, 0)            # Left
        elif action == 1:
            row = min(row + 1, self.grid_rows - 1)  # Down
        elif action == 2:
            col = min(col + 1, self.grid_cols - 1)  # Right
        elif action == 3:
            row = max(row - 1, 0)            # Up

        self.s = row * self.grid_cols + col
        tile = self.tiles[self.s]

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
