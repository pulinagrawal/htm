from __future__ import annotations

from statistics import fmean
from typing import Any, Callable, Optional
from core.HTM import ColumnField, Field, DUTY_CYCLE_PERIOD

class ValueField(ColumnField):
    """Standalone value estimator that operates on a ColumnField via composition.

    Holds a reference to a field and tracks per-cell TD values and eligibility
    traces without requiring the field to inherit from anything.

    Usage::

        d1 = ColumnField(input_fields=[layer5], ...)
        d1_tracker = ValueTracker(d1, weight_fn=lambda cell: int(cell.active))
        # each timestep:
        d1_tracker.update_values(reward=reward)
        layer5.compute(learn=True, td_error=d1_tracker.avg_error)

    ``trace`` is computed as an exponential moving sum:

        trace_t = trace_decay * trace_(t-1) + value_t
    """

    def __init__(
        self,
        input_fields: List[Field],
        num_columns: int = 0,
        cells_per_column: int = 1,
        non_spatial: bool = False,
        non_temporal: bool = False,
        duty_cycle_period: int = DUTY_CYCLE_PERIOD,
        go_field: 'ValueField|None' = None,
        nogo_field: 'ValueField|None' = None,
        weight_fn: Optional[Callable[[Any], float]] = None,
    ) -> None:
        super().__init__(
            input_fields=input_fields,
            num_columns=num_columns,
            cells_per_column=cells_per_column,
            non_spatial=non_spatial,
            non_temporal=non_temporal,
            duty_cycle_period=duty_cycle_period,
            go_field=go_field,
            nogo_field=nogo_field,
        ) 
        self._field = self
        self.values = [0.0] * len(self._field.cells)
        self.traces = [0.0] * len(self._field.cells)
        self.td_learning_rate = 0.5
        self.td_discount = 0.95
        self.trace_decay = 0.6
        self.avg_error = 0.0
        self._weight_fn = weight_fn or ValueField._default_weight

    @staticmethod
    def _default_weight(cell: Any) -> float:
        """State-value weight (Eq. 5.2).

        A successfully predicted active cell weighs 10, a bursting active cell
        weighs 1, an inactive cell weighs 0. A cell is "correctly predicted"
        when it was predictive on the previous step and is active now
        (``prev_predictive and active``); a cell active without a prior
        prediction is bursting.
        """
        if cell.prev_predictive and cell.active:  # Correct prediction
            return 10.0
        if not cell.prev_predictive and cell.active:  # False positive
            return 1.0
        return 0.0

    def avg_value(self) -> float:
        """Weighted state value (Eq. 5.2).

        ``(sum_i Value_i * Weight_i) / n`` where ``n`` is the size of the
        neural activation (the number of active cells), not the total cell
        count. Inactive cells contribute nothing to numerator or denominator.
        """
        numerator = 0.0
        n = 0
        for cell, v in zip(self._field.cells, self.values):
            active = cell.active 
            if active:
                n += 1
                numerator += self._weight_fn(cell) * v
        return numerator / n if n else 0.0

    def calculate_avg_error(self, reward: float) -> None:
        """TD error (Eq. 5.3).

        ``AvgError = mean_i (R + gamma * AvgValue - Value_i)`` over the
        *previous-state* neurons, where ``Value_i`` is each previous neuron's
        stored value and ``AvgValue`` is the current (weighted) state value.
        """
        avg_value = self.avg_value()
        prev_values = [v for cell, v in zip(self._field.cells, self.values)
                       if cell.prev_active]
        if not prev_values:
            self.avg_error = 0.0
            return
        self.avg_error = fmean(
            reward + self.td_discount * avg_value - value
            for value in prev_values
        )

    def update_values(self, reward) -> None:
        """Per-cycle reward-layer update (Alg. 9 lines 15-19).

        Order: decay traces -> compute error -> refresh traces -> update values.
        """
        self.decay_traces()                       # Eq. 5.1 (x gamma * lambda)
        self.calculate_avg_error(reward=reward)   # Eq. 5.2 & 5.3
        self.refresh_traces()                     # Eq. 5.4 (prev-active -> 1)
        for i in range(len(self.values)):         # Eq. 5.5
            self.values[i] += self.td_learning_rate * self.avg_error * self.traces[i]

    def decay_traces(self) -> None:
        """Eq. 5.1: decay every eligibility trace by gamma * lambda."""
        decay = self.td_discount * self.trace_decay
        for i in range(len(self.traces)):
            self.traces[i] *= decay

    def refresh_traces(self) -> None:
        """Eq. 5.4: reset the traces of previously-active neurons to 1."""
        for i, cell in enumerate(self._field.cells):
            if cell.prev_active:
                self.traces[i] = 1.0
    