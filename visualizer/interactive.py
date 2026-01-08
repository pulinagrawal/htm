from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np
import random

from building_blocks import (
    ColumnField,
    InputField,
    CONNECTED_PERM,
)
from rdse import RDSEParameters

from visualizer.simulator import SimulationConfig


@dataclass
class StepSnapshot:
    """Compact summary of a single ColumnField update."""

    step: int
    input_value: float
    active_columns: int
    bursting_columns: int
    predictive_columns: int


@dataclass
class ManualValueFeed:
    """Stateful feed that mirrors a UI-controlled value."""

    value: float = 0.0

    def generator(self) -> Iterator[float]:
        while True:
            yield self.value

    def update(self, new_value: float) -> None:
        self.value = new_value


@dataclass
class InteractiveContext:
    config: SimulationConfig
    input_field: InputField
    column_field: ColumnField
    feed_iter: Iterator[float]
    learn: bool
    feed_label: str
    manual_feed: Optional[ManualValueFeed] = None
    step: int = 0
    last_snapshot: Optional[StepSnapshot] = None


def sine_value_generator(cycle_length: int) -> Iterator[float]:
    """Yield a repeating sine wave, mirroring tests/test_real_data.py."""
    sine_cycle = np.sin(np.linspace(0, 2 * np.pi, cycle_length, endpoint=False))
    while True:
        for value in sine_cycle:
            yield float(value)


def create_interactive_context(
    config: SimulationConfig,
    feed_iter: Iterator[float],
    learn: bool,
    feed_label: str,
    manual_feed: Optional[ManualValueFeed] = None,
) -> InteractiveContext:
    random.seed(config.python_seed)
    np.random.seed(config.python_seed)
    input_field, column_field = _build_fields(config)
    return InteractiveContext(
        config=config,
        input_field=input_field,
        column_field=column_field,
        feed_iter=feed_iter,
        learn=learn,
        feed_label=feed_label,
        manual_feed=manual_feed,
    )


def advance_context(ctx: InteractiveContext, steps: int) -> StepSnapshot:
    if steps <= 0:
        raise ValueError("steps must be positive")
    snapshot: Optional[StepSnapshot] = None
    for _ in range(steps):
        value = next(ctx.feed_iter)
        ctx.input_field.encode(value)
        ctx.column_field.compute(learn=ctx.learn)
        ctx.step += 1
        snapshot = StepSnapshot(
            step=ctx.step,
            input_value=value,
            active_columns=len(ctx.column_field.active_columns),
            bursting_columns=len(ctx.column_field.bursting_columns),
            predictive_columns=_count_predictive_columns(ctx.column_field),
        )
    ctx.last_snapshot = snapshot
    if snapshot is None:
        raise RuntimeError("Failed to advance context")
    return snapshot


def column_state_codes(column_field: ColumnField) -> List[int]:
    """Encode each column into a discrete state for plotting."""
    codes: List[int] = []
    for column in column_field.columns:
        if column.bursting:
            codes.append(3)
        elif column.active:
            codes.append(2)
        elif any(cell.predictive for cell in column.cells):
            codes.append(1)
        else:
            codes.append(0)
    return codes


def describe_column_cells(column_field: ColumnField, column_index: int) -> List[dict]:
    column = column_field.columns[column_index]
    rows: List[dict] = []
    for idx, cell in enumerate(column.cells):
        rows.append(
            {
                "cell": idx,
                "active": cell.active,
                "predictive": cell.predictive,
                "learning": cell.learning,
                "segments": len(cell.segments),
            }
        )
    return rows


def describe_cell_segments(column_field: ColumnField, column_index: int, cell_index: int) -> List[dict]:
    column = column_field.columns[column_index]
    cell = column.cells[cell_index]
    rows: List[dict] = []
    for idx, segment in enumerate(cell.segments):
        rows.append(
            {
                "segment": idx,
                "synapses": len(segment.synapses),
                "connected_synapses": sum(1 for syn in segment.synapses if syn.permanence >= CONNECTED_PERM),
                "active_synapses": sum(1 for syn in segment.synapses if syn.source_cell and syn.source_cell.active),
                "prev_learning_synapses": len(segment.get_synapses_to_prev_learning_cells()),
                "active": segment.active,
                "learning": segment.learning,
            }
        )
    return rows


def describe_segment_synapses(
    column_field: ColumnField,
    column_index: int,
    cell_index: int,
    segment_index: int,
) -> List[dict]:
    column = column_field.columns[column_index]
    cell = column.cells[cell_index]
    if segment_index >= len(cell.segments):
        return []
    segment = cell.segments[segment_index]
    column_lookup = {id(col): idx for idx, col in enumerate(column_field.columns)}
    rows: List[dict] = []
    for idx, syn in enumerate(segment.synapses):
        source_column = None
        source_cell = None
        if syn.source_cell is not None and syn.source_cell.parent_column is not None:
            source_column = column_lookup.get(id(syn.source_cell.parent_column))
            source_cell = syn.source_cell.parent_column.cells.index(syn.source_cell)
        rows.append(
            {
                "synapse": idx,
                "permanence": syn.permanence,
                "connected": syn.permanence >= CONNECTED_PERM,
                "source_column": source_column,
                "source_cell": source_cell,
                "source_prev_active": bool(syn.source_cell.prev_active) if syn.source_cell else False,
                "source_prev_learning": bool(syn.source_cell.prev_learning) if syn.source_cell else False,
            }
        )
    return rows


def _build_fields(config: SimulationConfig) -> tuple[InputField, ColumnField]:
    params = RDSEParameters(
        size=config.num_columns,
        active_bits=config.active_bits,
        sparsity=0.0,
        radius=0.0,
        resolution=config.resolution,
        category=False,
        seed=config.rdse_seed,
    )
    input_field = InputField(size=config.num_columns, rdse_params=params)
    column_field = ColumnField(
        input_fields=[input_field],
        non_spatial=True,
        num_columns=config.num_columns,
        cells_per_column=config.cells_per_column,
    )
    return input_field, column_field


def _count_predictive_columns(column_field: ColumnField) -> int:
    return sum(1 for column in column_field.columns if any(cell.predictive for cell in column.cells))
