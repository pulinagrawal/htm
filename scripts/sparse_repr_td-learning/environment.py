"""Self-contained deterministic 4x4 FrozenLake — *tabular model only*.

No gym, no rendering, no HTM. Just the minimal MDP we need to:
  - step a fixed policy and feed (s, r, s') transitions to the TD learner, and
  - compute the analytic ground-truth value function V^pi by policy evaluation,
    so the sanity test can check that the thesis TD update converges to the
    *correct* Bellman values (not just to *some* fixed point).

Map / dynamics are byte-for-byte the ones in ``src/continuous_frozen_lake.py``:
actions 0=Left, 1=Down, 2=Right, 3=Up, clamped at the grid edge; reward by tile.
Holes (H) and the goal (G) are terminal here (episodic) so the discounted value
gradient is finite and textbook-clean.
"""

from __future__ import annotations

import random

GRID_ROWS = 4
GRID_COLS = 4
N_STATES = GRID_ROWS * GRID_COLS
N_ACTIONS = 4

TILES = [
    "S", "F", "F", "F",
    "F", "H", "F", "H",
    "F", "F", "F", "H",
    "H", "F", "F", "G",
]

START_STATE = 0
GOAL_STATE = TILES.index("G")          # 15
TERMINAL = {s for s, t in enumerate(TILES) if t in ("G", "H")}

# (goal_r, hole_r, step_r) — same schedule as the regular_vs_continuous experiment.
RewardSchedule = tuple[float, float, float]
DEFAULT_REWARDS: RewardSchedule = (10.0, -10.0, -1.0)

# Optimal deterministic path S -> G (Down,Down,Right,Down,Right,Right).
OPTIMAL_PATH = [0, 4, 8, 9, 13, 14, 15]
# Optimal action to take *out of* each on-path state (last entry G is terminal).
OPTIMAL_ACTIONS = {0: 1, 4: 1, 8: 2, 9: 1, 13: 2, 14: 2}
# In the *continuous* (non-terminal) setting the goal is non-absorbing: Right (2) keeps
# the agent on tile 15 (bottom-right corner, clamped), so "camp on the goal" — banking
# +10 every step — is the optimal action there. This is the attractor behaviour the
# regular_vs_continuous_fzlake experiment documents.
GOAL_CAMP_ACTION = 2


def next_state(s: int, a: int) -> int:
    """Deterministic transition (edge-clamped). Mirrors ContinuousFrozenLake.step."""
    row, col = divmod(s, GRID_COLS)
    if a == 0:
        col = max(col - 1, 0)               # Left
    elif a == 1:
        row = min(row + 1, GRID_ROWS - 1)   # Down
    elif a == 2:
        col = min(col + 1, GRID_COLS - 1)   # Right
    elif a == 3:
        row = max(row - 1, 0)               # Up
    return row * GRID_COLS + col


def reward_of(s: int, schedule: RewardSchedule = DEFAULT_REWARDS) -> float:
    """Reward for *landing on* state ``s``."""
    goal_r, hole_r, step_r = schedule
    tile = TILES[s]
    if tile == "G":
        return goal_r
    if tile == "H":
        return hole_r
    return step_r


class FrozenLakeModel:
    """Episodic deterministic FrozenLake. Terminal on G/H, reset to S."""

    def __init__(self, schedule: RewardSchedule = DEFAULT_REWARDS) -> None:
        self.schedule = schedule
        self.s = START_STATE

    def reset(self) -> int:
        self.s = START_STATE
        return self.s

    def step(self, a: int) -> tuple[int, float, bool]:
        """Take action ``a``; return (next_state, reward, done)."""
        s2 = next_state(self.s, a)
        r = reward_of(s2, self.schedule)
        done = s2 in TERMINAL
        self.s = s2
        return s2, r, done


# -- Fixed policies -------------------------------------------------------------

def _optimal_actions(continuous: bool) -> dict[int, int]:
    """Defined optimal actions. In continuous mode the goal also has one (camp)."""
    if continuous:
        return {**OPTIMAL_ACTIONS, GOAL_STATE: GOAL_CAMP_ACTION}
    return OPTIMAL_ACTIONS


def epsilon_soft_optimal(epsilon: float, rng: random.Random, continuous: bool = False):
    """A fixed stochastic policy: optimal action w.p. 1-eps, uniform-random else.

    eps=0 -> pure optimal path (only the 7 path states are ever visited).
    eps>0 -> covers the whole grid, while the optimal action still dominates.

    In ``continuous`` mode the goal gets a defined "camp" action so the agent stays
    on the (non-absorbing) goal instead of wandering off it.
    """
    opt = _optimal_actions(continuous)

    def policy(s: int) -> int:
        if rng.random() < epsilon or s not in opt:
            return rng.randrange(N_ACTIONS)
        return opt[s]

    return policy


def action_distribution(epsilon: float, continuous: bool = False) -> list[list[float]]:
    """P(a | s) table for the eps-soft-optimal policy — used by policy evaluation."""
    opt = _optimal_actions(continuous)
    dist = []
    for s in range(N_STATES):
        if s in opt:
            p = [epsilon / N_ACTIONS] * N_ACTIONS
            p[opt[s]] += 1.0 - epsilon
        else:
            p = [1.0 / N_ACTIONS] * N_ACTIONS  # uniform where no optimal action defined
        dist.append(p)
    return dist


def policy_evaluation(
    epsilon: float,
    gamma: float,
    continuous: bool = False,
    schedule: RewardSchedule = DEFAULT_REWARDS,
    theta: float = 1e-12,
    max_iter: int = 1_000_000,
) -> list[float]:
    """Exact V^pi for the eps-soft-optimal policy. Ground truth for the TD learner.

    ``continuous=False`` (episodic): G/H are terminal, V(terminal)=0, bootstrap cut.
    ``continuous=True``: no terminals at all — every state bootstraps, the goal is a
    non-absorbing attractor (V(goal) -> goal_r/(1-gamma)). Same deterministic dynamics.
    """
    dist = action_distribution(epsilon, continuous)
    V = [0.0] * N_STATES
    for _ in range(max_iter):
        delta = 0.0
        for s in range(N_STATES):
            if not continuous and s in TERMINAL:
                continue  # V(terminal) = 0 by episodic convention
            v_new = 0.0
            for a in range(N_ACTIONS):
                pa = dist[s][a]
                if pa == 0.0:
                    continue
                s2 = next_state(s, a)
                r = reward_of(s2, schedule)
                # Episodic: cut bootstrap on landing in a terminal. Continuous: never.
                bootstrap = 0.0 if (not continuous and s2 in TERMINAL) else gamma * V[s2]
                v_new += pa * (r + bootstrap)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    return V


def make_continuous_env(seed: int, schedule: RewardSchedule = DEFAULT_REWARDS):
    """Build the *real* ``src/continuous_frozen_lake.py::ContinuousFrozenLake`` — the
    same non-terminal, reset-free env used by ``regular_vs_continuous_fzlake/``.

    A gym env is required only because ContinuousFrozenLake reads its observation/action
    spaces from one (and would use it for an optional renderer, which we don't enable).
    """
    import sys
    import warnings
    from pathlib import Path

    warnings.filterwarnings("ignore", message=".*render method without specifying.*")
    src_dir = Path(__file__).resolve().parents[2] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    import gymnasium as gym
    from continuous_frozen_lake import ContinuousFrozenLake

    genv = gym.make("FrozenLake-v1", is_slippery=False)
    genv.reset(seed=seed)
    return ContinuousFrozenLake(reward_schedule=schedule, env=genv)
