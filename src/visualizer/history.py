"""State snapshot and history buffer for step-back functionality."""

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HTMSnapshot:
    """Captures HTM state for a single timestep."""
    timestep: int
    inputs: dict[str, Any]
    predictions: dict[str, Any]

    # Per input field: list of active cell indices
    input_active: dict[str, list[int]] = field(default_factory=dict)
    input_predictive: dict[str, list[int]] = field(default_factory=dict)

    # Per column field: cell states as (col_idx, cell_idx) tuples
    column_active_cells: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    column_winner_cells: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    column_predictive_cells: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    column_bursting: dict[str, list[int]] = field(default_factory=dict)
    column_active_cols: dict[str, list[int]] = field(default_factory=dict)

    # Go/NoGo depolarization: (col_idx, cell_idx) tuples
    column_go_cells: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    column_nogo_cells: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    # Apical segment counts
    num_go_segments: dict[str, int] = field(default_factory=dict)
    num_nogo_segments: dict[str, int] = field(default_factory=dict)

    # ValueField TD state
    td_avg_error: dict[str, float] = field(default_factory=dict)
    td_avg_value: dict[str, float] = field(default_factory=dict)

    # OutputField activation probabilities and decoded action
    output_probabilities: dict[str, list[float]] = field(default_factory=dict)
    output_decoded: dict[str, dict] = field(default_factory=dict)

    # Stats
    num_segments: dict[str, int] = field(default_factory=dict)
    num_synapses: dict[str, int] = field(default_factory=dict)


class History:
    """Circular buffer of HTM snapshots for step-back navigation."""

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self._buffer: list[HTMSnapshot] = []
        self._position: int = -1  # current viewing position

    def capture(self, brain, timestep: int, inputs: dict, predictions: dict) -> HTMSnapshot:
        """Capture current brain state into a snapshot."""
        snap = HTMSnapshot(timestep=timestep, inputs=inputs, predictions=predictions)

        for name, f in brain.all_input_fields.items():
            snap.input_active[name] = [i for i, c in enumerate(f.cells) if c.active]
            snap.input_predictive[name] = [i for i, c in enumerate(f.cells) if c.predictive]

        for name, f in brain.all_column_fields.items():
            snap.column_active_cells[name] = [
                (ci, ji) for ci, col in enumerate(f.columns)
                for ji, cell in enumerate(col.cells) if cell.active
            ]
            snap.column_winner_cells[name] = [
                (ci, ji) for ci, col in enumerate(f.columns)
                for ji, cell in enumerate(col.cells) if cell.winner
            ]
            snap.column_predictive_cells[name] = [
                (ci, ji) for ci, col in enumerate(f.columns)
                for ji, cell in enumerate(col.cells) if cell.predictive
            ]
            snap.column_bursting[name] = [
                ci for ci, col in enumerate(f.columns) if col.bursting
            ]
            snap.column_active_cols[name] = [
                ci for ci, col in enumerate(f.columns) if col.active
            ]

            # Go/NoGo depolarization states
            snap.column_go_cells[name] = [
                (ci, ji) for ci, col in enumerate(f.columns)
                for ji, cell in enumerate(col.cells)
                if hasattr(cell, 'go_depolarized') and cell.go_depolarized
            ]
            snap.column_nogo_cells[name] = [
                (ci, ji) for ci, col in enumerate(f.columns)
                for ji, cell in enumerate(col.cells)
                if hasattr(cell, 'nogo_depolarized') and cell.nogo_depolarized
            ]

            # Apical segment counts
            snap.num_go_segments[name] = sum(
                len(cell.go_segments) for col in f.columns for cell in col.cells
                if hasattr(cell, 'go_segments')
            )
            snap.num_nogo_segments[name] = sum(
                len(cell.nogo_segments) for col in f.columns for cell in col.cells
                if hasattr(cell, 'nogo_segments')
            )

            snap.num_segments[name] = sum(
                len(cell.distal_segments) for col in f.columns for cell in col.cells
            )
            snap.num_synapses[name] = sum(
                len(seg.synapses) for col in f.columns
                for cell in col.cells for seg in cell.distal_segments
            )

        # ValueField TD state
        if hasattr(brain, '_value_fields'):
            for name, vf in brain._value_fields.items():
                snap.td_avg_error[name] = vf.avg_error
                snap.td_avg_value[name] = vf.avg_value()

        # OutputField activation probabilities and decoded action
        if hasattr(brain, '_output_fields'):
            for name, of in brain._output_fields.items():
                if hasattr(of, 'activation_probabilities'):
                    snap.output_probabilities[name] = of.activation_probabilities()
                if hasattr(of, 'decode'):
                    try:
                        snap.output_decoded[name] = of.decode()
                    except Exception:
                        snap.output_decoded[name] = {"value": None, "confidence": None}

        # Trim future if we stepped back then captured new
        if self._position < len(self._buffer) - 1:
            self._buffer = self._buffer[:self._position + 1]

        self._buffer.append(snap)
        if len(self._buffer) > self.max_size:
            self._buffer.pop(0)
        self._position = len(self._buffer) - 1
        return snap

    @property
    def current(self) -> HTMSnapshot | None:
        if not self._buffer:
            return None
        return self._buffer[self._position]

    def step_back(self) -> HTMSnapshot | None:
        if self._position > 0:
            self._position -= 1
        return self.current

    def step_forward(self) -> HTMSnapshot | None:
        if self._position < len(self._buffer) - 1:
            self._position += 1
        return self.current

    @property
    def can_step_back(self) -> bool:
        return self._position > 0

    @property
    def can_step_forward(self) -> bool:
        return self._position < len(self._buffer) - 1

    def __len__(self):
        return len(self._buffer)
