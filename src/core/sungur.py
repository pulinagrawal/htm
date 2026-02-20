from __future__ import annotations

from statistics import fmean
from typing import Any

class ValueFieldMixin:
    """Mixin that adds per-cell value and trace computation to a field.

    The mixin only assumes the host class exposes a ``cells`` iterable.
    Place this mixin before ``Field`` in MRO:

        class MyField(ValueFieldMixin, Field):
            ...

    ``trace`` is computed as an exponential moving sum:

        trace_t = trace_decay * trace_(t-1) + value_t
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.values = [0]*len(self.cells)
        self.traces = [0]*len(self.cells) 
        self.td_learning_rate = 0.1
        self.td_discount = 0.9
        self.trace_decay = 0.9
        self._avg_error = 0.0

    @staticmethod
    def weight(cell: Any) -> float:
        """Default scalar weight for a cell based on HTM-like boolean states."""
        if cell.prev_predictive and cell.active: # Correct prediction
            return 10
        if not cell.prev_predictive and cell.active: # False positive
            return 1
        return 0

    def avg_value(self) -> float:
        """Compute value for a cell based on its state and the weight function."""
        return fmean(self.weight(cell)*value 
                     for cell, value in zip(self.cells, self.values))

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
        for i, cell in enumerate(self.cells):
            self.values[i] += self.td_learning_rate * self.avg_error * self.traces[i]
        self.decay_traces()

    def decay_traces(self) -> None:
        """Update trace values for all cells based on current cell states."""
        for i, cell in enumerate(self.cells):
            if cell.active:
                self.traces[i] = 1
            else:
                self.traces[i] *= self.td_discount*self.trace_decay
    
    def compute_intrinsic_reward(self) -> float:
        """Compute an intrinsic reward signal based on value prediction error."""
        raise NotImplementedError("Needs Implementation")

