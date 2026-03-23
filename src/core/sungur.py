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
        self.td_learning_rate = 0.3
        self.td_discount = 0.5
        self.trace_decay = 0.6
        self.avg_error = 0.0
        self._weight_fn = weight_fn or ValueField._default_weight

    @staticmethod
    def _default_weight(cell: Any) -> float:
        """Default weight for a cell based on HTM-like boolean states."""
        if cell.prev_predictive and cell.active:  # Correct prediction
            return 10.0
        if not cell.prev_predictive and cell.active:  # False positive
            return 1.0
        return 0.0

    def avg_value(self) -> float:
        """Weighted average of per-cell value estimates."""
        return fmean(
            self._weight_fn(cell) * v
            for cell, v in zip(self._field.cells, self.values)
        )

    def calculate_avg_error(self, reward: float) -> float:
        avg_value = self.avg_value()
        # TODO: consider using per neuron errors instead of averaging.
        # This is a design choice: TD learning typically uses a single scalar error signal
        # But the thesis does it differently (refer to: https://claude.ai/share/72e97d45-7428-4185-b0fe-11052852f9be)
        self.avg_error = fmean(reward + self.td_discount*avg_value-value
                                for value in self.values)

    def update_values(self, reward) -> None:
        """Update value estimates for all cells based on current states."""
        self.calculate_avg_error(reward=reward)
        for i, cell in enumerate(self._field.cells):
            self.values[i] += self.td_learning_rate * self.avg_error * self.traces[i]
        self.decay_traces()

    def decay_traces(self) -> None:
        """Update trace values for all cells based on current cell states."""
        for i, cell in enumerate(self._field.cells):
            if cell.active:
                self.traces[i] = 1
            else:
                self.traces[i] *= self.td_discount*self.trace_decay
    