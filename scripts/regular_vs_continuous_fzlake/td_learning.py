"""Shared tabular SARSA core used for both the episodic and continuous experiments.

Deliberately minimal — a plain Q-table with epsilon-greedy action selection and the
canonical one-step SARSA update. The *only* thing the two experiments do differently is
whether they pass ``done=True`` (terminal bootstrap cutoff) to ``update``.
"""

from __future__ import annotations

import random
from collections import defaultdict


class TabularSARSA:
    """Tabular on-policy TD(0) control (SARSA).

    Q is a ``defaultdict(float)`` so unseen ``(state, action)`` pairs start at 0 — the
    same optimistic-relative-to-negative-rewards initialization the existing
    ``agents/td_agent.py`` relies on.
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        rng: random.Random | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = rng or random.Random()
        self.Q: dict[tuple[int, int], float] = defaultdict(float)

    def _argmax(self, state: int) -> int:
        """Greedy action with random tie-breaking (avoids first-action bias)."""
        qs = [self.Q[(state, a)] for a in range(self.n_actions)]
        best = max(qs)
        candidates = [a for a, q in enumerate(qs) if q == best]
        return self.rng.choice(candidates)

    def choose(self, state: int) -> int:
        """Epsilon-greedy action selection (behavior policy)."""
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        return self._argmax(state)

    def greedy(self, state: int) -> int:
        """Pure greedy action (target policy, used at evaluation time)."""
        return self._argmax(state)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int | None,
        done: bool,
    ) -> float:
        """One SARSA update. Returns the (signed) TD error.

        ``done=True`` applies the terminal bootstrap cutoff (target = reward only),
        which is what makes the episodic update differ from the continuous one.
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[(next_state, next_action)]
        td = target - self.Q[(state, action)]
        self.Q[(state, action)] += self.alpha * td
        return td

    def value(self, state: int) -> float:
        """State value V(s) = max_a Q(s, a)."""
        return max(self.Q[(state, a)] for a in range(self.n_actions))
