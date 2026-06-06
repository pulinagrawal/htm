"""Thesis control logic (D1/D2 Go/No-Go reward circuitry + voluntary action) on the
stripped sparse-SDR substrate — no HTM SP/TM, but the *control* architecture is faithful.

Maps to the reference (`HTM_TD_Agent_Implementation_Reference.md`):

  - §8 reward circuitry: two striatal populations **D1 (Go)** and **D2 (No-Go)**, each a
    per-neuron value over the SDR, learned by TD(λ) (Eqs 5.1–5.5). They differ by the
    **sign of the TD error** they learn from (D1 ← positive, D2 ← negative) — this is the
    fix for **B7** (in the full HTM agent D1/D2 collapsed to identical values, so the
    Go−NoGo motor signal was zero everywhere and the policy never moved).
  - §3.2 / §5.3.3 distal prediction: the agent learns a one-step model M[sdr][a] =
    (next_sdr, reward, done) from experience — the functional stand-in for the L4/L2/L5
    distal Temporal Memory that predicts the reachable next state per motor action.
  - §9 voluntary action: a candidate next state that is both **distally predicted**
    (model-known) and **apically depolarized** (Go/No-Go value signal) drives the motor
    neurons — excitatory from D1 (Go), inhibitory from D2 (No-Go).
  - §6 / §9.2 motor: n_actions × 3 motor neurons, drive + **base random excitation**,
    **top-3 WTA**; the action with the most active motor neurons is executed.

Key identity that makes the split sound: with V_go updated on δ≥0 and V_nogo on δ<0,
the combined value V = V_go − V_nogo obeys ΔV = α·δ·trace in *both* branches, i.e. it is
exactly a standard TD(λ) critic — while Go and No-Go remain distinct populations. The net
motor drive for an action then equals the true one-step value:

    drive(a) = [r + γ·V_go(s')] − [γ·V_nogo(s')] = r + γ·V(s') = Q(s, a).

(Reward is included in the lookahead because in the *episodic* task the goal is terminal
with V(goal)=0; without the immediate reward the agent would never step onto it.)
"""

from __future__ import annotations

import numpy as np

Sdr = tuple[int, ...]


class GoNoGoActor:
    """D1/D2 actor-critic over a sparse code with thesis-style voluntary action."""

    def __init__(
        self,
        n_neurons: int,
        n_actions: int,
        alpha: float = 0.5,           # TD_LEARNING_RATE
        gamma: float = 0.95,          # TD_DISCOUNT_RATE
        lam: float = 0.6,             # TD_TRACE_DECAY (lambda)
        base_excitation: float = 4.0,  # motor base excitation (exploration magnitude)
        epsilon: float = 0.0,          # explicit random-action rate (see act())
        neurons_per_action: int = 3,   # §6: 3 motor neurons per action
        wta_k: int = 3,                # §6: top-3 winner-take-all
        rng: np.random.Generator | None = None,
    ) -> None:
        self.alpha, self.gamma, self.lam = alpha, gamma, lam
        self.n_actions = n_actions
        self.base_excitation = base_excitation
        self.epsilon = epsilon
        self.npa = neurons_per_action
        self.wta_k = wta_k
        self.rng = rng or np.random.default_rng()

        # D1 (Go) and D2 (No-Go) per-neuron values + shared eligibility traces.
        self.go = np.zeros(n_neurons)
        self.nogo = np.zeros(n_neurons)
        self.traces = np.zeros(n_neurons)

        # Learned one-step model: sdr -> action -> (next_sdr, reward, done).
        self.model: dict[Sdr, dict[int, tuple[Sdr, float, bool]]] = {}

    # -- value read-outs (Eq 5.2 with all-burst weight 1) -----------------------
    def _mean(self, arr: np.ndarray, sdr: Sdr | None) -> float:
        if not sdr:
            return 0.0
        idx = np.asarray(sdr, dtype=int)
        return float(arr[idx].mean())

    def value_go(self, sdr: Sdr | None) -> float:
        return self._mean(self.go, sdr)

    def value_nogo(self, sdr: Sdr | None) -> float:
        return self._mean(self.nogo, sdr)

    def value(self, sdr: Sdr | None) -> float:
        """Combined state value V = V_go − V_nogo (a standard TD(λ) critic)."""
        return self.value_go(sdr) - self.value_nogo(sdr)

    # -- one-step action values via the learned model ---------------------------
    def _drive(self, sdr: Sdr) -> tuple[np.ndarray, np.ndarray]:
        """Per-action net motor drive Q(s,a) and a mask of model-known actions.

        Voluntary action requires a *distally predicted* next state (model known).
        Unknown actions get drive 0 → only base excitation acts on them (exploration).
        """
        drive = np.zeros(self.n_actions)
        known = np.zeros(self.n_actions, dtype=bool)
        transitions = self.model.get(sdr, {})
        for a, (next_sdr, reward, done) in transitions.items():
            v_next = 0.0 if done else self.value(next_sdr)
            # Go excites by (r + γ·V_go), No-Go inhibits by (γ·V_nogo); net = r + γ·V.
            drive[a] = reward + self.gamma * v_next
            known[a] = True
        return drive, known

    # -- §9.2 voluntary excitation + §6 motor WTA -------------------------------
    def act(self, sdr: Sdr) -> int:
        """Behaviour policy: Go/No-Go drive + base random excitation, top-3 WTA.

        Early on (values ≈ 0) base excitation dominates → random exploration; as the
        Go/No-Go values grow the drive dominates → goal-directed behaviour.

        ``epsilon`` is an explicit random-action rate — the controllable form of "base
        excitation produces random motion". It is needed in the *reset-free* setting,
        where the high-value goal is a non-absorbing attractor: a fixed base-excitation
        magnitude cannot both lose to a ~200-scale drive near the goal and still randomise
        the small-valued start region, so the agent camps and the start region starves
        (the same coverage problem `regular_vs_continuous_fzlake/` fixed with ε≈0.6).
        """
        if self.epsilon and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        drive, _known = self._drive(sdr)
        # n_actions × npa motor neurons; each carries its action's drive + base noise.
        neuron_action = np.repeat(np.arange(self.n_actions), self.npa)
        excitation = drive[neuron_action] + self.base_excitation * self.rng.random(
            self.n_actions * self.npa
        )
        # Top-k WTA, then the action with the most winning motor neurons.
        winners = np.argpartition(excitation, -self.wta_k)[-self.wta_k:]
        votes = np.bincount(neuron_action[winners], minlength=self.n_actions)
        best = np.flatnonzero(votes == votes.max())
        return int(self.rng.choice(best))

    def greedy(self, sdr: Sdr) -> int:
        """Target policy (evaluation): pick the highest-drive *known* action.

        Falls back to a random action when the state has no learned transitions.
        """
        drive, known = self._drive(sdr)
        if not known.any():
            return int(self.rng.integers(self.n_actions))
        masked = np.where(known, drive, -np.inf)
        best = np.flatnonzero(masked == masked.max())
        return int(self.rng.choice(best))

    # -- learning ---------------------------------------------------------------
    def observe(self, sdr: Sdr, action: int, reward: float,
                next_sdr: Sdr | None, done: bool) -> float:
        """Record the transition in the model and apply the sign-split TD(λ) update.

        ``next_sdr`` is the real encoded next state (even when terminal) so the model
        can learn that an action reaches the goal/hole; ``done`` cuts the value bootstrap.
        Returns the signed TD error.
        """
        # Learn the one-step model (distal prediction stand-in).
        self.model.setdefault(sdr, {})[action] = (next_sdr or (), reward, done)

        # TD(λ), thesis order (Alg 9 ll.15–19): decay → error → refresh → update.
        self.traces *= self.gamma * self.lam                      # Eq 5.1
        v_next = 0.0 if done else self.value(next_sdr)            # Eq 5.2
        v_prev = self.value(sdr)
        error = reward + self.gamma * v_next - v_prev            # Eq 5.3

        idx = np.asarray(sdr, dtype=int)
        self.traces[idx] = 1.0                                    # Eq 5.4 (replacing)
        if error >= 0.0:                                          # B7 sign-split + Eq 5.5
            self.go += self.alpha * error * self.traces           # D1 ← positive error
        else:
            self.nogo += self.alpha * (-error) * self.traces      # D2 ← negative error
        return error

    def end_episode(self) -> None:
        """Clear eligibility traces at an episode boundary (no cross-episode credit)."""
        self.traces[:] = 0.0
