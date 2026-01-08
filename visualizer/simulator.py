from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from statistics import fmean
from typing import Dict, List

import numpy as np
import random

from building_blocks import (
    ColumnField,
    InputField,
    CONNECTED_PERM,
)
from rdse import RDSEParameters


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters that govern the HTM training run."""

    num_columns: int = 512
    cells_per_column: int = 16
    active_bits: int = 16
    resolution: float = 0.01
    cycle_length: int = 64
    total_steps: int = 600
    python_seed: int = 42
    rdse_seed: int = 5
    noise_std: float = 0.0
    evaluation_cycles: int = 1


@dataclass
class SimulationResult:
    config: SimulationConfig
    training_values: List[float]
    burst_counts: List[int]
    active_columns: List[int]
    predictive_columns: List[int]
    evaluation_values: List[float]
    evaluation_bursts: List[int]
    stats_text: str
    column_duty_cycles: List[float]
    cell_duty_cycles: List[float]
    permanence_values: List[float]
    metrics: Dict[str, float]


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Train a ColumnField on a sine wave and return diagnostic signals."""

    _seed_rngs(config)
    input_field, column_field = _build_fields(config)
    sine_cycle = _build_sine_cycle(config)

    training_values: List[float] = []
    burst_counts: List[int] = []
    active_columns: List[int] = []
    predictive_columns: List[int] = []

    for step in range(config.total_steps):
        value = _sample_value(step, sine_cycle, config)
        training_values.append(value)
        input_field.encode(value)
        column_field.compute()
        burst_counts.append(len(column_field.bursting_columns))
        active_columns.append(len(column_field.active_columns))
        predictive_columns.append(_count_predictive_columns(column_field))

    column_field.clear_states()

    evaluation_values, evaluation_bursts = _evaluate(column_field, input_field, sine_cycle, config)

    snapshot = _collect_snapshot(column_field)
    stats_text = _render_stats(column_field)
    metrics = _build_metrics(burst_counts, predictive_columns, evaluation_bursts, snapshot, config)

    return SimulationResult(
        config=config,
        training_values=training_values,
        burst_counts=burst_counts,
        active_columns=active_columns,
        predictive_columns=predictive_columns,
        evaluation_values=evaluation_values,
        evaluation_bursts=evaluation_bursts,
        stats_text=stats_text,
        column_duty_cycles=snapshot["column_duty_cycles"],
        cell_duty_cycles=snapshot["cell_duty_cycles"],
        permanence_values=snapshot["permanence_values"],
        metrics=metrics,
    )


def _seed_rngs(config: SimulationConfig) -> None:
    random.seed(config.python_seed)
    np.random.seed(config.python_seed)


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


def _build_sine_cycle(config: SimulationConfig) -> np.ndarray:
    return np.sin(np.linspace(0, 2 * np.pi, config.cycle_length, endpoint=False))


def _sample_value(step: int, sine_cycle: np.ndarray, config: SimulationConfig) -> float:
    value = float(sine_cycle[step % config.cycle_length])
    if config.noise_std > 0.0:
        value += float(np.random.normal(loc=0.0, scale=config.noise_std))
    return value


def _count_predictive_columns(column_field: ColumnField) -> int:
    predictive_count = 0
    for column in column_field.columns:
        predictive_count += 1 if any(cell.predictive for cell in column.cells) else 0
    return predictive_count


def _evaluate(
    column_field: ColumnField,
    input_field: InputField,
    sine_cycle: np.ndarray,
    config: SimulationConfig,
) -> tuple[List[float], List[int]]:
    evaluation_steps = config.cycle_length * max(1, config.evaluation_cycles)
    evaluation_values: List[float] = []
    evaluation_bursts: List[int] = []
    for step in range(evaluation_steps):
        value = float(sine_cycle[step % config.cycle_length])
        evaluation_values.append(value)
        input_field.encode(value)
        column_field.compute(learn=False)
        evaluation_bursts.append(len(column_field.bursting_columns))
    return evaluation_values, evaluation_bursts


def _collect_snapshot(column_field: ColumnField) -> Dict[str, List[float] | float]:
    segments = [segment for cell in column_field.cells for segment in cell.segments]
    synapses = [syn for segment in segments for syn in segment.synapses]
    permanences = [float(syn.permanence) for syn in synapses]
    connected_synapses = sum(1 for permanence in permanences if permanence >= CONNECTED_PERM)
    total_synapses = len(permanences)
    connected_ratio = (connected_synapses / total_synapses) if total_synapses else 0.0
    column_duty_cycles = [float(column.active_duty_cycle) for column in column_field.columns]
    cell_duty_cycles = [float(cell.active_duty_cycle) for cell in column_field.cells]
    return {
        "permanence_values": permanences,
        "column_duty_cycles": column_duty_cycles,
        "cell_duty_cycles": cell_duty_cycles,
        "connected_ratio": connected_ratio,
    }


def _render_stats(column_field: ColumnField) -> str:
    buffer = StringIO()
    with redirect_stdout(buffer):
        column_field.print_stats()
    return buffer.getvalue()


def _build_metrics(
    burst_counts: List[int],
    predictive_columns: List[int],
    evaluation_bursts: List[int],
    snapshot: Dict[str, List[float] | float],
    config: SimulationConfig,
) -> Dict[str, float]:
    final_burst = float(burst_counts[-1]) if burst_counts else 0.0
    quarter_len = max(1, len(burst_counts) // 4)
    mean_final_quarter = float(fmean(burst_counts[-quarter_len:])) if burst_counts else 0.0
    predictive_share = (
        float(fmean(predictive_columns)) / float(config.num_columns)
        if predictive_columns
        else 0.0
    )
    burst_free_ratio = (
        evaluation_bursts.count(0) / float(len(evaluation_bursts))
        if evaluation_bursts
        else 0.0
    )
    connected_ratio = float(snapshot.get("connected_ratio", 0.0))
    return {
        "final_burst_count": final_burst,
        "mean_burst_last_quarter": mean_final_quarter,
        "predictive_column_share": predictive_share,
        "burst_free_ratio": burst_free_ratio,
        "connected_synapse_ratio": connected_ratio,
    }
