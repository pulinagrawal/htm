"""Generic NxN FrozenLake gridworld — the *larger* env for the overlap experiment.

``environment.py`` hard-codes the 4x4 map (16 states); to let a *spatial* (grid
distance) encoder actually show its overlap structure we need more room between
tiles, so this module loads an arbitrary map string (default: the standard gym
**8x8** FrozenLake, 64 states) and derives everything generically:

  - the deterministic edge-clamped dynamics (identical rule to
    ``src/continuous_frozen_lake.py`` / ``environment.py``);
  - the **optimal policy + optimal path** by value iteration (no hand-coded path,
    since the map is now arbitrary);
  - the exact ground-truth V^pi for an eps-soft-optimal policy by policy
    evaluation — the analytic target the TD learner must match.

Same reward schedule and conventions as ``environment.py`` so results are
directly comparable to the 4x4 sanity test.
"""

from __future__ import annotations

import random

# Standard gym maps (start top-left S, goal bottom-right G, H = hole).
MAPS: dict[str, list[str]] = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}

# (goal_r, hole_r, step_r) — same schedule as environment.py / regular_vs_continuous.
RewardSchedule = tuple[float, float, float]
DEFAULT_REWARDS: RewardSchedule = (10.0, -10.0, -1.0)


class GridWorld:
    """Episodic deterministic NxN FrozenLake derived from a map string."""

    def __init__(self, map_name: str = "8x8",
                 schedule: RewardSchedule = DEFAULT_REWARDS) -> None:
        rows = MAPS[map_name]
        self.map_name = map_name
        self.rows = len(rows)
        self.cols = len(rows[0])
        self.tiles = [c for row in rows for c in row]
        self.n_states = self.rows * self.cols
        self.n_actions = 4
        self.schedule = schedule
        self.start_state = self.tiles.index("S")
        self.goal_state = self.tiles.index("G")
        self.terminal = {s for s, t in enumerate(self.tiles) if t in ("G", "H")}
        self.s = self.start_state

    # -- coordinates ------------------------------------------------------------
    def coords(self, s: int) -> tuple[int, int]:
        """(row, col) of state index ``s``."""
        return divmod(s, self.cols)

    # -- dynamics (edge-clamped; mirrors ContinuousFrozenLake.step) -------------
    def next_state(self, s: int, a: int) -> int:
        row, col = divmod(s, self.cols)
        if a == 0:
            col = max(col - 1, 0)                # Left
        elif a == 1:
            row = min(row + 1, self.rows - 1)    # Down
        elif a == 2:
            col = min(col + 1, self.cols - 1)    # Right
        elif a == 3:
            row = max(row - 1, 0)                # Up
        return row * self.cols + col

    def reward_of(self, s: int) -> float:
        goal_r, hole_r, step_r = self.schedule
        tile = self.tiles[s]
        if tile == "G":
            return goal_r
        if tile == "H":
            return hole_r
        return step_r

    # -- episodic env API -------------------------------------------------------
    def reset(self) -> int:
        self.s = self.start_state
        return self.s

    def step(self, a: int) -> tuple[int, float, bool]:
        s2 = self.next_state(self.s, a)
        r = self.reward_of(s2)
        done = s2 in self.terminal
        self.s = s2
        return s2, r, done

    # -- planning: optimal policy / path via value iteration --------------------
    def optimal_actions(self, gamma: float, theta: float = 1e-12) -> dict[int, int]:
        """argmax-action per non-terminal state for the *optimal* policy (Q*)."""
        V = [0.0] * self.n_states
        while True:
            delta = 0.0
            for s in range(self.n_states):
                if s in self.terminal:
                    continue
                best = max(
                    self.reward_of(self.next_state(s, a))
                    + (0.0 if self.next_state(s, a) in self.terminal
                       else gamma * V[self.next_state(s, a)])
                    for a in range(self.n_actions)
                )
                delta = max(delta, abs(best - V[s]))
                V[s] = best
            if delta < theta:
                break
        opt: dict[int, int] = {}
        for s in range(self.n_states):
            if s in self.terminal:
                continue
            opt[s] = max(
                range(self.n_actions),
                key=lambda a: self.reward_of(self.next_state(s, a))
                + (0.0 if self.next_state(s, a) in self.terminal
                   else gamma * V[self.next_state(s, a)]),
            )
        return opt

    def optimal_path(self, gamma: float) -> list[int]:
        """Greedy roll-out S -> G under the optimal policy (cycle-guarded)."""
        opt = self.optimal_actions(gamma)
        path = [self.start_state]
        s = self.start_state
        seen = {s}
        while s != self.goal_state and s in opt:
            s = self.next_state(s, opt[s])
            if s in seen:                # safety: avoid an infinite loop
                break
            path.append(s)
            seen.add(s)
        return path


# -- Fixed eps-soft-optimal policy + analytic V^pi (ground truth) ---------------

def epsilon_soft_optimal(env: GridWorld, epsilon: float, gamma: float,
                         rng: random.Random):
    """Optimal action w.p. 1-eps, uniform-random else (covers the whole grid)."""
    opt = env.optimal_actions(gamma)

    def policy(s: int) -> int:
        if rng.random() < epsilon or s not in opt:
            return rng.randrange(env.n_actions)
        return opt[s]

    return policy


def _action_distribution(env: GridWorld, epsilon: float, gamma: float):
    opt = env.optimal_actions(gamma)
    dist = []
    for s in range(env.n_states):
        if s in opt:
            p = [epsilon / env.n_actions] * env.n_actions
            p[opt[s]] += 1.0 - epsilon
        else:
            p = [1.0 / env.n_actions] * env.n_actions
        dist.append(p)
    return dist


def policy_evaluation(env: GridWorld, epsilon: float, gamma: float,
                      theta: float = 1e-12, max_iter: int = 1_000_000) -> list[float]:
    """Exact V^pi for the eps-soft-optimal policy (episodic: V(terminal)=0)."""
    dist = _action_distribution(env, epsilon, gamma)
    V = [0.0] * env.n_states
    for _ in range(max_iter):
        delta = 0.0
        for s in range(env.n_states):
            if s in env.terminal:
                continue
            v_new = 0.0
            for a in range(env.n_actions):
                pa = dist[s][a]
                if pa == 0.0:
                    continue
                s2 = env.next_state(s, a)
                r = env.reward_of(s2)
                bootstrap = 0.0 if s2 in env.terminal else gamma * V[s2]
                v_new += pa * (r + bootstrap)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    return V
