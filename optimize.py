#!/usr/bin/env python3
"""Parameter sweep for sine-wave prediction error.

This script mirrors manual_test.test_sine_wave_bursting_columns_converge while
optimizing the ColumnField hyperparameters that govern temporal memory growth.

The output JSON is intentionally schema-light:
- Each trial stores a free-form ``params`` mapping.
- Each trial stores a free-form ``metrics`` mapping.

This keeps the optimizer and plotting scripts resilient as you add/remove
optimized parameters.
"""
from __future__ import annotations

import argparse
import json
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from tqdm import tqdm

import HTM as bb
from HTM import CONNECTED_PERM, ColumnField, Field, InputField
from rdse import RDSEParameters, RandomDistributedScalarEncoder


@dataclass(frozen=True)
class SineConfig:
    """Replica of the configuration used by test_sine_wave_bursting_columns_converge."""

    num_columns: int = 512
    cells_per_column: int = 16
    sparsity: float = 0.02
    resolution: float = 0.001
    cycle_length: int = 64
    total_steps: int = 1_000
    evaluation_cycles: int = 1
    rdse_seed: int = 5
    rng_seed: int = 42
    missing_prediction_penalty: float = 2.0


@dataclass(frozen=True)
class TrialResult:
    """Evaluation summary for a single hyper-parameter combination."""

    params: dict[str, float]
    metrics: dict[str, float | int | bool]


@contextmanager
def override_tm_constants(params: dict[str, float]) -> Iterator[None]:
    """Temporarily override ColumnField-related constants.

    Any numeric entry in ``params`` whose upper-cased name exists as an
    attribute on the HTM module will be overridden.
    """

    previous: dict[str, float] = {}
    overrides: dict[str, float] = {}
    for key, value in params.items():
        constant_name = key.upper()
        if not hasattr(bb, constant_name):
            continue
        if not isinstance(value, (int, float)):
            continue
        overrides[constant_name] = float(value)

    try:
        for name, value in overrides.items():
            previous[name] = getattr(bb, name)
            setattr(bb, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(bb, name, value)


def sine_cycle(length: int) -> np.ndarray:
    """Precompute a unit sine wave used throughout training and evaluation."""

    return np.sin(np.linspace(0.0, 2.0 * np.pi, length, endpoint=False))


def describe_structure(column_field: ColumnField) -> tuple[int, int, int]:
    """Return (total_segments, total_synapses, connected_synapses)."""

    segments = [segment for cell in column_field.cells for segment in cell.segments]
    synapses = [syn for segment in segments for syn in segment.synapses]
    connected_synapses = sum(1 for syn in synapses if float(syn.permanence) >= CONNECTED_PERM)
    return len(segments), len(synapses), connected_synapses


def duty_cycle_nonzero_share(values: Sequence[float], *, eps: float = 0.0) -> float:
    if not values:
        return 0.0
    nonzero = sum(1 for value in values if float(value) > eps)
    return float(nonzero / len(values))


def evaluate_trial(
    params: dict[str, float],
    config: SineConfig,
    sine_values: Sequence[float],
    *,
    burst_weight: float,
    segments_weight: float,
    synapses_weight: float,
    min_nonzero_duty_share: float,
    invalid_penalty: float,
) -> TrialResult:
    """Run the sine-wave test while overriding the requested constants."""

    random.seed(config.rng_seed)
    np.random.seed(config.rng_seed)

    rdse_params = RDSEParameters(
        size=config.num_columns,
        sparsity=config.sparsity,
        resolution=config.resolution,
        category=False,
        seed=config.rdse_seed,
    )

    with override_tm_constants(params):
        input_field = InputField(size=config.num_columns, rdse_params=rdse_params)
        column_field = ColumnField(
            input_fields=[input_field],
            non_spatial=True,
            num_columns=config.num_columns,
            cells_per_column=config.cells_per_column,
        )

        train_bursts: list[int] = []
        for step in range(config.total_steps):
            value = float(sine_values[step % config.cycle_length])
            input_field.encode(value)
            column_field.compute()
            train_bursts.append(len(column_field.bursting_columns))

        total_segments, total_synapses, connected_synapses = describe_structure(column_field)
        column_duty_share = duty_cycle_nonzero_share(
            [column.active_duty_cycle for column in column_field.columns]
        )
        cell_duty_share = duty_cycle_nonzero_share(
            [cell.active_duty_cycle for cell in column_field.cells]
        )

        evaluation_steps = config.cycle_length * config.evaluation_cycles
        errors: list[float] = []
        prediction_failures = 0
        evaluation_bursts: list[int] = []

        for step in range(evaluation_steps):
            target_value = float(sine_values[step % config.cycle_length])
            prediction_field = column_field.get_prediction()[0]
            predicted_value, _ = input_field.decode(prediction_field, 'predictive')
            if predicted_value is None:
                prediction_failures += 1
                abs_error = config.missing_prediction_penalty
            else:
                abs_error = abs(predicted_value - target_value)**2
            errors.append(abs_error)
            input_field.encode(target_value)
            column_field.compute(learn=False)
            evaluation_bursts.append(len(column_field.bursting_columns))

    mean_abs_error = float(np.mean(errors)) if errors else float("inf")
    max_abs_error = float(np.max(errors)) if errors else float("inf")
    avg_eval_bursting = float(np.mean(evaluation_bursts[1:])) if evaluation_bursts else 0.0

    train_max_initial = int(max(train_bursts[: min(10, len(train_bursts))])) if train_bursts else 0
    train_final = int(train_bursts[-1]) if train_bursts else 0

    total_cells = max(1, config.num_columns * config.cells_per_column)
    segments_per_cell = total_segments / total_cells
    synapses_per_cell = total_synapses / total_cells
    connected_ratio = (connected_synapses / total_synapses) if total_synapses else 0.0

    valid = not (
        column_duty_share < min_nonzero_duty_share and cell_duty_share < min_nonzero_duty_share
    )

    score = (
        mean_abs_error
        - burst_weight * avg_eval_bursting
        - segments_weight * segments_per_cell
        - synapses_weight * synapses_per_cell
    )
    if not valid:
        score += invalid_penalty

    metrics: dict[str, float | int | bool] = {
        "mean_abs_error": mean_abs_error,
        "max_abs_error": max_abs_error,
        "prediction_failures": int(prediction_failures),
        "avg_eval_bursting_columns": avg_eval_bursting,
        "train_max_initial_burst": int(train_max_initial),
        "train_final_burst": int(train_final),
        "total_segments": int(total_segments),
        "total_synapses": int(total_synapses),
        "connected_synapses": int(connected_synapses),
        "connected_synapse_ratio": float(connected_ratio),
        "column_duty_nonzero_share": float(column_duty_share),
        "cell_duty_nonzero_share": float(cell_duty_share),
        "valid": bool(valid),
        "score": float(score),
    }

    return TrialResult(params=params, metrics=metrics)


def generate_param_grid(args: argparse.Namespace) -> Iterator[dict[str, float]]:
    """Create a deterministic Cartesian product over the provided value lists."""

    for growth in args.growth_strengths:
        for max_syn in args.max_synapse_pcts:
            for activation in args.activation_threshold_pcts:
                for learning in args.learning_threshold_pcts:
                    for predicted_decrement in args.predicted_decrement_pcts:
                        for receptive_field in args.receptive_field_pcts:
                            yield {
                                "growth_strength": float(growth),
                                "max_synapse_pct": float(max_syn),
                                "activation_threshold_pct": float(activation),
                                "learning_threshold_pct": float(activation)*float(learning),
                                "predicted_decrement_pct": float(predicted_decrement),
                                "receptive_field_pct": float(receptive_field),
                            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep ColumnField hyperparameters to minimize sine-wave prediction error. "
            "Defaults mirror manual_test.test_sine_wave_bursting_columns_converge."
        )
    )
    parser.add_argument("--growth-strengths", type=float, nargs="+", default=[0.9, 0.5, 0.1])
    parser.add_argument("--max-synapse-pcts", type=float, nargs="+", default=[0.008])
    parser.add_argument("--activation-threshold-pcts", type=float, nargs="+", default=[.5, 0.8, 0.9])
    parser.add_argument("--learning-threshold-pcts", type=float, nargs="+", default=[0.5])
    parser.add_argument(
        "--predicted-decrement-pcts",
        dest="predicted_decrement_pcts",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 0.9],
    )
    parser.add_argument(
        "--receptive-field-pcts",
        dest="receptive_field_pcts",
        type=float,
        nargs="+",
        default=[0.1],
    )
    # Backwards-compat aliases
    parser.add_argument(
        "--predicted_decrement_pct",
        dest="predicted_decrement_pcts",
        type=float,
        nargs="+",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--receptive_field_pct",
        dest="receptive_field_pcts",
        type=float,
        nargs="+",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--num-columns", type=int, default=512)
    parser.add_argument("--cells-per-column", type=int, default=16)
    parser.add_argument("--total-steps", type=int, default=1_000)
    parser.add_argument("--cycle-length", type=int, default=64)
    parser.add_argument("--evaluation-cycles", type=int, default=1)
    parser.add_argument("--resolution", type=float, default=0.001)
    parser.add_argument("--sparsity", type=float, default=0.02)
    parser.add_argument("--rdse-seed", type=int, default=5)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prediction_error_results.json"),
        help="Destination JSON file for all trial metrics.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="How many best trials to print.")
    parser.add_argument("--burst-weight", type=float, default=0.05)
    parser.add_argument("--segments-weight", type=float, default=0.005)
    parser.add_argument("--synapses-weight", type=float, default=0.0001)
    parser.add_argument(
        "--min-duty-nonzero-share",
        type=float,
        default=0.1,
        help=(
            "Minimum share of columns/cells with duty_cycle > 0. "
            "If BOTH column and cell shares are below this threshold, the trial is invalid."
        ),
    )
    parser.add_argument(
        "--invalid-penalty",
        type=float,
        default=1_000.0,
        help="Penalty added to the score for invalid trials.",
    )
    return parser.parse_args()


def timestamped_path(path: Path) -> Path:
    """Append a datetime suffix so concurrent runs never clobber each other."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = path.suffix or ".json"
    return path.with_name(f"{path.stem}_{timestamp}{suffix}")


def main() -> None:
    args = parse_args()
    sine_values = sine_cycle(args.cycle_length).tolist()
    config = SineConfig(
        num_columns=args.num_columns,
        cells_per_column=args.cells_per_column,
        sparsity=args.sparsity,
        resolution=args.resolution,
        cycle_length=args.cycle_length,
        total_steps=args.total_steps,
        evaluation_cycles=args.evaluation_cycles,
        rdse_seed=args.rdse_seed,
        rng_seed=args.rng_seed,
    )

    trials: list[TrialResult] = []
    total_trials = (
        len(args.growth_strengths)
        * len(args.max_synapse_pcts)
        * len(args.activation_threshold_pcts)
        * len(args.learning_threshold_pcts)
        * len(args.predicted_decrement_pcts)
        * len(args.receptive_field_pcts)
    )

    for params in tqdm(generate_param_grid(args), total=total_trials, desc="Parameter sweep"):
        trials.append(
            evaluate_trial(
                params,
                config,
                sine_values,
                burst_weight=args.burst_weight,
                segments_weight=args.segments_weight,
                synapses_weight=args.synapses_weight,
                min_nonzero_duty_share=args.min_duty_nonzero_share,
                invalid_penalty=args.invalid_penalty,
            )
        )

    trials.sort(key=lambda trial: float(trial.metrics["score"]))
    best_trials = trials[: args.top_k]

    summary = {
        "config": asdict(config),
        "weights": {
            "burst_weight": args.burst_weight,
            "segments_weight": args.segments_weight,
            "synapses_weight": args.synapses_weight,
            "min_duty_nonzero_share": args.min_duty_nonzero_share,
            "invalid_penalty": args.invalid_penalty,
        },
        "results": [
            {
                "params": trial.params,
                "metrics": trial.metrics,
            }
            for trial in trials
        ],
    }

    output_path = timestamped_path(args.output)
    output_path.write_text(json.dumps(summary, indent=2))

    print("\nTop trials:")
    for trial in best_trials:
        print(json.dumps(
            {
                "params": trial.params,
                "metrics": trial.metrics,
            },
            indent=2,
        ))

    print(f"\nFull results written to {output_path}")


if __name__ == "__main__":
    main()
