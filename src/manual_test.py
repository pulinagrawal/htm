#!/usr/bin/env python3
"""Sine-wave regression test and evaluator for HTM parameter sweeps."""
from __future__ import annotations

import random
import numpy as np
from dataclasses import dataclass
from typing import Any
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.sungur import ColumnField, InputField
from brain import Brain
from encoder_layer.rdse import RDSEParameters


@dataclass(frozen=True)
class SineConfig:
    num_columns: int = 512
    cells_per_column: int = 16
    resolution: float = 0.001
    cycle_length: int = 64
    rdse_seed: int = 5
    total_steps: int = 1_000
    rng_seed: int = 42
    missing_prediction_penalty: float = 2.0


def _resolve_config(config: dict[str, Any] | SineConfig | None) -> SineConfig:
    if config is None:
        return SineConfig()
    if isinstance(config, SineConfig):
        return config
    return SineConfig(**config)


def _seed_rng(config: SineConfig) -> None:
    np.random.seed(config.rng_seed)
    random.seed(config.rng_seed)


def _build_sine_brain(config: SineConfig) -> tuple[Brain, list[float]]:
    rdse_params = RDSEParameters(
        size=config.num_columns,
        sparsity=0.02,
        resolution=config.resolution,
        category=False,
        seed=config.rdse_seed,
    )
    input_field = InputField(size=config.num_columns, encoder_params=rdse_params)
    column_field = ColumnField(
        input_fields=[input_field],
        non_spatial=True,
        num_columns=config.num_columns,
        cells_per_column=config.cells_per_column,
    )
    sine_cycle = np.sin(
        np.linspace(0, 2 * np.pi, config.cycle_length, endpoint=False)
    ).tolist()
    brain = Brain(
        {
            "sine_input": input_field,
            "column_field": column_field,
        }
    )
    return brain, sine_cycle


def _train_sine(
    brain: Brain,
    sine_cycle: list[float],
    config: SineConfig,
) -> list[int]:
    burst_counts = []
    for step in tqdm(range(config.total_steps), desc="Training"):
        value = sine_cycle[step % config.cycle_length]
        brain.step({"sine_input": value})
        burst_counts.append(len(brain.column_field.bursting_columns))
    print("Burst counts over time:", burst_counts)
    return burst_counts


def _evaluate_sine(
    brain: Brain,
    sine_cycle: list[float],
    config: SineConfig,
) -> tuple[list[float], list[int], int]:
    evaluation_bursts = []
    errors = []
    prediction_failures = 0
    for value in sine_cycle:
        prediction = brain.prediction()["sine_input"]
        if prediction is None:
            prediction_failures += 1
            errors.append(config.missing_prediction_penalty)
        else:
            errors.append(abs(value - prediction) ** 2)
        brain.step({"sine_input": value}, learn=False)
        evaluation_bursts.append(len(brain.column_field.bursting_columns))
    print("Evaluation burst counts:", evaluation_bursts)
    print("Evaluation errors:", errors)
    return errors, evaluation_bursts, prediction_failures


def _summarize_sine_metrics(
    errors: list[float],
    evaluation_bursts: list[int],
    prediction_failures: int,
    burst_counts: list[int],
    params: dict[str, float],
) -> dict[str, Any]:
    mae = sum(errors[1:]) / len(errors[1:])
    score = float(mae)
    return {
        "mean_abs_error": float(mae),
        "prediction_failures": int(prediction_failures),
        "avg_eval_bursting_columns": float(
            sum(evaluation_bursts[1:]) / max(1, len(evaluation_bursts[1:]))
        ),
        "train_final_burst": int(burst_counts[-1]) if burst_counts else 0,
        "score": score,
        "params_used": params,
    }


def evaluate_sine_wave(
    params: dict[str, float] | None = None,
    config: dict[str, Any] | SineConfig | None = None,
) -> dict[str, Any]:
    """Evaluate sine-wave prediction and return metrics for optimization."""
    params = params or {}
    config_obj = _resolve_config(config)
    _seed_rng(config_obj)
    brain, sine_cycle = _build_sine_brain(config_obj)
    burst_counts = _train_sine(brain, sine_cycle, config_obj)
    errors, evaluation_bursts, prediction_failures = _evaluate_sine(
        brain, sine_cycle, config_obj
    )
    brain.column_field.print_stats()
    return _summarize_sine_metrics(
        errors, evaluation_bursts, prediction_failures, burst_counts, params
    )


def test_sine_wave_bursting_columns_converge() -> None:
    """Test ColumnField bursts converge to zero on a learned sine-driven sequence."""
    metrics = evaluate_sine_wave()
    print("Mean Absolute Error of predictions:", metrics["mean_abs_error"])
    print("Prediction failures:", metrics["prediction_failures"])
    print("Evaluation bursting columns:", metrics["avg_eval_bursting_columns"])

if __name__ == "__main__":
    test_sine_wave_bursting_columns_converge()

""""
Stats 

learning =.5*num_synapses
ColumnField statistics:
  Columns: 512 | Cells: 8192 | Segments: 114 | Synapses: 16731
  +------------------------+--------------------+----------+----------+
  | Metric                 |   Mean ± Std      |      Min |      Max |
  +------------------------+--------------------+----------+----------+
  | Segments per cell     |     0.01 ± 0.33    |        0 |       16 |
  | Synapses per segment  |   146.76 ± 134.99  |       16 |      496 |
  | Permanence            |    0.145 ± 0.197   |    0.000 |    1.000 |
  +------------------------+--------------------+----------+----------+
  Connected synapses (>= 0.5): 492 (2.9% of all synapses)

learning = 5
ColumnField statistics:
  Columns: 512 | Cells: 8192 | Segments: 51 | Synapses: 11263
  +------------------------+--------------------+----------+----------+
  | Metric                 |   Mean ± Std      |      Min |      Max |
  +------------------------+--------------------+----------+----------+
  | Segments per cell     |     0.01 ± 0.10    |        0 |        2 |
  | Synapses per segment  |   220.84 ± 191.64  |        0 |      559 |
  | Permanence            |    0.047 ± 0.199   |    0.000 |    1.000 |
  +------------------------+--------------------+----------+----------+
  Connected synapses (>= 0.5): 524 (4.7% of all synapses)

learning = .1*num_synapses
ColumnField statistics:
  Columns: 512 | Cells: 8192 | Segments: 38 | Synapses: 11327
  +------------------------+--------------------+----------+----------+
  | Metric                 |   Mean ± Std       |      Min |      Max |
  +------------------------+--------------------+----------+----------+
  | Segments per cell      |     0.00 ± 0.08    |        0 |        4 |
  | Synapses per segment   |   298.08 ± 160.36  |       16 |      559 |
  | Permanence             |    0.048 ± 0.199   |    0.000 |    1.000 |
  +------------------------+--------------------+----------+----------+
  Connected synapses (>= 0.5): 524 (4.6% of all synapses)

activation threshold = .005*max_synapses
max_new_synapse_count=.1*field
ColumnField statistics:
  Columns: 512 | Cells: 8192 | Segments: 38 | Synapses: 11102
  +------------------------+--------------------+----------+----------+
  | Metric                 |   Mean ± Std      |      Min |      Max |
  +------------------------+--------------------+----------+----------+
  | Segments per cell     |     0.00 ± 0.08    |        0 |        4 |
  | Synapses per segment  |   292.16 ± 157.54  |       16 |      559 |
  | Permanence            |    0.049 ± 0.200   |    0.000 |    1.000 |
  +------------------------+--------------------+----------+----------+
  Connected synapses (>= 0.5): 524 (4.7% of all synapses)
"""