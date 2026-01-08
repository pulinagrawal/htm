#!/usr/bin/env python3
"""Parameter sweep for sine-wave prediction error.

This script mirrors manual_test.test_sine_wave_bursting_columns_converge while
optimizing the ColumnField hyperparameters that govern temporal memory growth.
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
from HTM import ColumnField, Field, InputField
from rdse import RDSEParameters, RandomDistributedScalarEncoder


@dataclass(frozen=True)
class HyperParameters:
    """Tunables sourced from building_blocks constants."""

    growth_strength: float
    max_synapse_pct: float
    activation_threshold_pct: float
    learning_threshold_pct: float


@dataclass(frozen=True)
class SineConfig:
    """Replica of the configuration used by test_sine_wave_bursting_columns_converge."""

    num_columns: int = 1024
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

    params: HyperParameters
    mean_abs_error: float
    max_abs_error: float
    prediction_failures: int
    avg_eval_bursting_columns: float
    train_max_initial_burst: int
    train_final_burst: int
    score: float


@contextmanager
def override_tm_constants(params: HyperParameters) -> Iterator[None]:
    """Temporarily override ColumnField-related constants."""

    overrides = {
        "GROWTH_STRENGTH": params.growth_strength,
        "MAX_SYNAPSE_PCT": params.max_synapse_pct,
        "ACTIVATION_THRESHOLD_PCT": params.activation_threshold_pct,
        "LEARNING_THRESHOLD_PCT": params.learning_threshold_pct,
    }
    previous: dict[str, float] = {}
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


def decode_prediction(prediction_field: Field, encoder: InputField, candidates: Sequence[float]):
    """Convert predictive column activity into a decoded scalar."""

    bit_vector = [1 if cell.predictive else 0 for cell in prediction_field.cells]
    if not any(bit_vector):
        return None, 0.0
    return RandomDistributedScalarEncoder.decode(encoder, bit_vector, candidates)


def evaluate_trial(params: HyperParameters, config: SineConfig, sine_values: Sequence[float]) -> TrialResult:
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

        evaluation_steps = config.cycle_length * config.evaluation_cycles
        errors: list[float] = []
        prediction_failures = 0
        evaluation_bursts: list[int] = []

        for step in range(evaluation_steps):
            target_value = float(sine_values[step % config.cycle_length])
            prediction_field = column_field.get_prediction()[0]
            predicted_value, _ = decode_prediction(prediction_field, input_field, sine_values)
            if predicted_value is None:
                prediction_failures += 1
                abs_error = config.missing_prediction_penalty
            else:
                abs_error = abs(predicted_value - target_value)
            errors.append(abs_error)
            input_field.encode(target_value)
            column_field.compute(learn=False)
            evaluation_bursts.append(len(column_field.bursting_columns))

    mean_abs_error = float(np.mean(errors)) if errors else float("inf")
    max_abs_error = float(np.max(errors)) if errors else float("inf")
    avg_eval_bursting = float(np.mean(evaluation_bursts)) if evaluation_bursts else 0.0

    train_max_initial = int(max(train_bursts[: min(10, len(train_bursts))])) if train_bursts else 0
    train_final = int(train_bursts[-1]) if train_bursts else 0

    score = mean_abs_error + 0.05 * avg_eval_bursting + 0.01 * train_final

    return TrialResult(
        params=params,
        mean_abs_error=mean_abs_error,
        max_abs_error=max_abs_error,
        prediction_failures=prediction_failures,
        avg_eval_bursting_columns=avg_eval_bursting,
        train_max_initial_burst=train_max_initial,
        train_final_burst=train_final,
        score=score,
    )


def generate_param_grid(args: argparse.Namespace) -> Iterator[HyperParameters]:
    """Create a deterministic Cartesian product over the provided value lists."""

    for growth in args.growth_strengths:
        for max_syn in args.max_synapse_pcts:
            for activation in args.activation_threshold_pcts:
                for learning in args.learning_threshold_pcts:
                    yield HyperParameters(
                        growth_strength=growth,
                        max_synapse_pct=max_syn,
                        activation_threshold_pct=activation,
                        learning_threshold_pct=activation*learning,
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep ColumnField hyperparameters to minimize sine-wave prediction error. "
            "Defaults mirror manual_test.test_sine_wave_bursting_columns_converge."
        )
    )
    parser.add_argument("--growth-strengths", type=float, nargs="+", default=[0.9])
    parser.add_argument("--max-synapse-pcts", type=float, nargs="+", default=[0.02])
    parser.add_argument("--activation-threshold-pcts", type=float, nargs="+", default=[0.1, 0.5, 0.8])
    parser.add_argument("--learning-threshold-pcts", type=float, nargs="+", default=[0.9, 0.5, 0.1])
    parser.add_argument("--num-columns", type=int, default=1024)
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
    )

    for params in tqdm(generate_param_grid(args), total=total_trials, desc="Parameter sweep"):
        trials.append(evaluate_trial(params, config, sine_values))

    trials.sort(key=lambda trial: trial.score)
    best_trials = trials[: args.top_k]

    summary = {
        "config": asdict(config),
        "results": [
            {
                "params": asdict(trial.params),
                "metrics": {
                    "mean_abs_error": trial.mean_abs_error,
                    "max_abs_error": trial.max_abs_error,
                    "prediction_failures": trial.prediction_failures,
                    "avg_eval_bursting_columns": trial.avg_eval_bursting_columns,
                    "train_max_initial_burst": trial.train_max_initial_burst,
                    "train_final_burst": trial.train_final_burst,
                    "score": trial.score,
                },
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
                "params": asdict(trial.params),
                "mean_abs_error": trial.mean_abs_error,
                "max_abs_error": trial.max_abs_error,
                "prediction_failures": trial.prediction_failures,
                "avg_eval_bursting_columns": trial.avg_eval_bursting_columns,
                "train_final_burst": trial.train_final_burst,
                "score": trial.score,
            },
            indent=2,
        ))

    print(f"\nFull results written to {output_path}")


if __name__ == "__main__":
    main()
