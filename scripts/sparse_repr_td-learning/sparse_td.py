"""Thesis reward-circuitry value learner (Eqs 5.1-5.5), stripped of all HTM.

This is a line-for-line port of ``src/core/sungur.py::ValueField`` reduced to its
TD essence: a per-neuron stored ``value`` and ``trace``, a weighted-mean state
value, the TD error, replacing eligibility traces, and the neuron-value update.

What is kept (identical to the thesis / ValueField):
  - Eq 5.1  DecayTraces:        trace_i *= gamma * lambda
  - Eq 5.2  CalculateAvgValue:  V(s) = (sum_i value_i * weight_i) / n_active
  - Eq 5.3  CalculateAvgError:  err = mean over prev-active of (R + gamma*V(s') - value_i)
  - Eq 5.4  RefreshTraces:      trace_i = 1 for previously-active neurons (replacing)
  - Eq 5.5  UpdateStateValues:  value_i += alpha * err * trace_i
  - Per-cycle order (Alg 9 ll.15-19): decay -> error -> refresh -> update.

What is dropped (the HTM stack this sanity test deliberately removes):
  - SP/TM, distal/apical prediction, D1/D2 wiring, voluntary action.
  - The predicted-vs-burst weight (10 vs 1): with no Temporal Memory there is no
    prediction, so *every* active neuron is "bursting" -> weight 1. AvgValue is
    then the plain mean of active-neuron values. (Set ``predicted_weight`` only if
    you reintroduce a prediction signal.)

The state is supplied directly as an SDR (tuple of active-neuron indices), so the
learner never sees the environment or any HTM machinery — only sparse codes.
"""

from __future__ import annotations

import numpy as np


class ThesisTDValue:
    """Eqs 5.1-5.5 over a flat array of neuron values + eligibility traces."""

    def __init__(
        self,
        n_neurons: int,
        alpha: float = 0.5,    # TD_LEARNING_RATE (thesis default)
        gamma: float = 0.95,   # TD_DISCOUNT_RATE
        lam: float = 0.6,      # TD_TRACE_DECAY (lambda)
        burst_weight: float = 1.0,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.burst_weight = burst_weight
        self.values = np.zeros(n_neurons)
        self.traces = np.zeros(n_neurons)
        self.last_error = 0.0

    # -- Eq 5.2 -----------------------------------------------------------------
    def value(self, sdr) -> float:
        """Weighted-mean state value over the active neurons (Eq 5.2).

        No prediction signal here, so weight_i = burst_weight for all active
        neurons -> V(s) = mean(values[active]). A terminal/empty SDR has V = 0
        (episodic no-bootstrap convention, matching policy_evaluation).
        """
        if sdr is None or len(sdr) == 0:
            return 0.0
        idx = np.asarray(sdr, dtype=int)
        return float(self.burst_weight * self.values[idx].sum() / idx.size)

    # -- Eqs 5.1, 5.3, 5.4, 5.5 (Alg 9 ll.15-19 order) --------------------------
    def update(self, prev_sdr, reward: float, next_sdr, done: bool) -> float:
        """One reward-circuitry cycle for the transition prev_sdr --reward--> next_sdr.

        Returns the (signed) TD error. ``done`` cuts the bootstrap (terminal
        next-state value = 0), the episodic convention used by the ground-truth
        policy evaluation.
        """
        # Eq 5.1 — decay every trace by gamma * lambda.
        self.traces *= self.gamma * self.lam

        # Eq 5.2 & 5.3 — current state value, then TD error over previous neurons.
        avg_value = 0.0 if done else self.value(next_sdr)
        prev_idx = np.asarray(prev_sdr, dtype=int)
        if prev_idx.size:
            # mean_i (R + gamma*V(s') - value_i) = R + gamma*V(s') - mean(prev values)
            error = reward + self.gamma * avg_value - float(self.values[prev_idx].mean())
        else:
            error = 0.0
        self.last_error = error

        # Eq 5.4 — replacing trace: previously-active neurons -> 1.
        if prev_idx.size:
            self.traces[prev_idx] = 1.0

        # Eq 5.5 — update every neuron's value by alpha * error * trace.
        self.values += self.alpha * error * self.traces
        return error

    def reset_traces(self) -> None:
        """Clear eligibility traces at an episode boundary (no cross-episode credit)."""
        self.traces[:] = 0.0
