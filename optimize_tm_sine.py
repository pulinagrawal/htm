#!/usr/bin/env python3
"""Hyperparameter search to satisfy the sine-wave bursting regression test."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, List
from tqdm import tqdm
import random

import numpy as np

from building_blocks import ColumnField, InputField
from rdse import RandomDistributedScalarEncoder as RDSE, RDSEParameters


@dataclass(frozen=True)
class SearchConfig:
    """Bundle all tunable knobs for the search."""

    num_columns: int
    cells_per_column: int
    activation_threshold: int
    learning_threshold: int
    max_new_synapse_count: int
    segment_growth_speed: int
    learning_rate: float
    active_bits: int
    resolution: float
    cycle_length: int
    rdse_seed: int


def build_search_space() -> List[SearchConfig]:
    """Create a deterministic Cartesian product search space."""
    num_columns = [1024]
    cells_per_column = [16]
    activation_threshold = [1, 3, 5]
    learning_threshold = [3, 5, 10]
    max_new_synapses = [16]
    segment_growth_speed = [1]
    learning_rate = [1.0]
    active_bits = [16]
    resolution = [0.1]
    cycle_length = [64]
    rdse_seed = [10000]

    configs: List[SearchConfig] = []
    for params in product(
        num_columns,
        cells_per_column,
        activation_threshold,
        learning_threshold,
        max_new_synapses,
        segment_growth_speed,
        learning_rate,
        active_bits,
        resolution,
        cycle_length,
        rdse_seed,
    ):
        configs.append(SearchConfig(*params))
    return configs


def sine_cycle(cycle_length: int) -> np.ndarray:
    """Precompute one sine cycle to avoid recomputing per iteration."""
    return np.sin(np.linspace(0.0, 2.0 * np.pi, cycle_length, endpoint=False))


def encode_value(encoder: RDSE, value: float) -> List[int]:
    """Encode a scalar using the RDSE instance."""
    return encoder.encode(float(value))


def evaluate_config(
    cfg: SearchConfig,
    total_steps: int,
    evaluation_passes: int,
    rng_seed: int,
) -> dict:
    """Measure whether the configuration eliminates late bursting."""
    np.random.seed(rng_seed)

    rdse_params = RDSEParameters(
        size=cfg.num_columns,
        active_bits=cfg.active_bits,
        sparsity=0.0,
        radius=0.0,
        resolution=cfg.resolution,
        category=False,
        seed=cfg.rdse_seed,
    )
    encoder = InputField(size=cfg.num_columns, rdse_params=rdse_params)

    tm = TemporalMemoryLayer(
        input_fields=[encoder],
        num_columns=cfg.num_columns,
        cells_per_column=cfg.cells_per_column,
        non_spatial=True,
        activation_threshold=cfg.activation_threshold,
        learning_threshold=cfg.learning_threshold,
    )
    sine_values = sine_cycle(cfg.cycle_length)

    burst_counts: List[int] = []
    for step in range(total_steps):
        value = sine_values[step % cfg.cycle_length]
        encoded_bits = encoder.encode(value)
        tm.compute()
        burst_counts.append(len(tm.bursting_columns))

    print("Burst counts over time:", burst_counts)
    evaluation_bursts: List[int] = []
    for step in range(evaluation_passes):
        value = sine_values[step % cfg.cycle_length]
        encoded_bits = encoder.encode(value)
        tm.compute(learn=False)
        evaluation_bursts.append(len(tm.bursting_columns))

    max_initial = max(burst_counts[: min(10, len(burst_counts))])
    final_burst = burst_counts[-1]
    steady_state_clear = all(count == 0 for count in evaluation_bursts[1:])

    metrics = {
        "max_initial_bursts": int(max_initial),
        "final_burst": int(final_burst),
        "evaluation_burst_sum": int(sum(evaluation_bursts[1:])),
        "passes": max_initial > 0 and final_burst == 0 and steady_state_clear,
    }
    penalty = 0 if metrics["passes"] else 1
    metrics["score"] = (
        final_burst * 1_000
        + metrics["evaluation_burst_sum"] * 10
        + (0 if max_initial > 0 else 100_000)
        + penalty * 500_000
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Search for TemporalMemoryLayer + RDSE parameters that satisfy the "
            "sine-wave bursting regression test."
        )
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=200,
        help="Upper bound on the number of configurations to evaluate.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=1000,
        help="Number of sine samples to stream during the learning phase.",
    )
    parser.add_argument(
        "--evaluation-passes",
        type=int,
        default=64,
        help="Cycle repetitions used for the post-learning evaluation phase.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Seed controlling the order in which configurations are tested.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Seed forwarded to numpy before each trial for reproducibility.",
    )
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="Stop immediately once a passing configuration is found.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hyperparam_search_results.json"),
        help="Where to store the ranked trial results as JSON.",
    )
    args = parser.parse_args()

    configs = build_search_space()
    rng = np.random.default_rng(args.shuffle_seed)
    rng.shuffle(configs)

    results = []
    best_result = None

    for idx, cfg in enumerate(configs[: args.max_trials]):
        metrics = evaluate_config(
            cfg=cfg,
            total_steps=args.total_steps,
            evaluation_passes=args.evaluation_passes,
            rng_seed=args.rng_seed,
        )
        trial = {"config": asdict(cfg), "metrics": metrics}
        results.append(trial)

        if best_result is None or metrics["score"] < best_result["metrics"]["score"]:
            best_result = trial

        print(
            f"Trial {idx+1:03d}: score={metrics['score']:,} "
            f"passes={metrics['passes']} final={metrics['final_burst']}"
        )

        if metrics["passes"] and args.stop_on_success:
            break

    summary = {
        "best_result": best_result,
        "trials": len(results),
    }

    args.output.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    if best_result:
        print("\nBest configuration:")
        print(json.dumps(best_result, indent=2))
    else:
        print("\nNo passing configuration found; inspect the JSON log for details.")

def test_sine_wave_bursting_columns_converge():
        """Test ColumnField bursts converge to zero on a learned sine-driven sequence."""
        config = {
            "num_columns": 512,
            "cells_per_column": 16,
            "activation_threshold": 3,
            "learning_threshold": 5,
            "active_bits": 16,
            "resolution": 0.001,
            "cycle_length": 64,
            "rdse_seed": 5,
            "total_steps": 1000,
        }

        np.random.seed(42)
        random.seed(42)
        params = RDSEParameters(
            size=config["num_columns"],
            active_bits=config["active_bits"],
            sparsity=0.0,
            radius=0.0,
            resolution=config["resolution"],
            category=False,
            seed=config["rdse_seed"],
        )
        input_field = InputField(size=config["num_columns"], rdse_params=params)
        column_field = ColumnField(
            input_fields=[input_field],
            non_spatial=True,
            num_columns=config["num_columns"],
            cells_per_column=config["cells_per_column"],
        )

        sine_cycle = np.sin(
            np.linspace(0, 2 * np.pi, config["cycle_length"], endpoint=False)
        )
        burst_counts = []

        for step in tqdm(range(config["total_steps"])):
            value = sine_cycle[step % config["cycle_length"]]
            input_field.encode(value)
            column_field.compute()
            burst_counts.append(len(column_field.bursting_columns))

        assert max(
            burst_counts[:10]
        ) > 0, "Column field should burst before learning the sine-driven sequence."
        print("Burst counts over time:", burst_counts)

        column_field.print_stats()
        assert burst_counts[-1] == 0, "Bursting columns should converge to zero after 1000 sine inputs."

        evaluation_bursts = []

        for value in sine_cycle:
            input_fields = column_field.get_prediction()
            prediction, confidence = input_field.decode(input_fields[0])
            assert abs(prediction - value) < 0.1, f"Prediction {prediction} not close to actual {value}"
            input_field.encode(value)
            column_field.compute(learn=False)
            evaluation_bursts.append(len(column_field.bursting_columns))

        burst_tolerance_pct = .05
        assert (
            sum(count for count in evaluation_bursts)
            <= burst_tolerance_pct * len(evaluation_bursts)
        ), f"Expected tolerant bursting once the sine sequence is mastered, got {evaluation_bursts}"

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