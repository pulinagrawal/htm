#!/usr/bin/env python3
"""Generic parameter sweep for HTM temporal-memory constants.

This optimizer is intentionally model/data agnostic. It overrides HTM
constants, calls a user-provided evaluator, and stores whatever metrics the
evaluator returns.

The output JSON is intentionally schema-light:
- Each trial stores a free-form ``params`` mapping.
- Each trial stores a free-form ``metrics`` mapping.
"""
from __future__ import annotations

import argparse
import importlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
import core.HTM as bb


@dataclass(frozen=True)
class TrialResult:
    """Evaluation summary for a single hyper-parameter combination."""

    params: dict[str, float]
    metrics: dict[str, Any]


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
            "Sweep ColumnField hyperparameters while calling a user-supplied evaluator."
        )
    )
    parser.add_argument("--growth-strengths", type=float, nargs="+", default=[0.3, 0.5, 0.6])
    parser.add_argument("--max-synapse-pcts", type=float, nargs="+", default=[0.008])
    parser.add_argument(
        "--activation-threshold-pcts",
        type=float,
        nargs="+",
        default=[0.4, 0.5, 0.6],
    )
    parser.add_argument("--learning-threshold-pcts", type=float, nargs="+", default=[0.25])

    parser.add_argument(
        "--predicted-decrement-pcts",
        dest="predicted_decrement_pcts",
        type=float,
        nargs="+",
        default=[0.1],
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
    parser.add_argument(
        "--evaluator",
        type=str,
        default="src.hot_gym_model:evaluate_hot_gym",
        help="Module path and function name, e.g. src.manual_test:evaluate_sine_wave",
    )
    parser.add_argument(
        "--evaluator-config",
        type=str,
        default=None,
        help="JSON string or path to JSON file passed to the evaluator.",
    )
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


def load_evaluator(path: str) -> Callable[[dict[str, float], dict[str, Any]], dict[str, Any]]:
    if ":" not in path:
        raise ValueError("Evaluator must be in format 'module:function'.")
    module_path, func_name = path.split(":", 1)
    module = importlib.import_module(module_path)
    evaluator = getattr(module, func_name, None)
    if not callable(evaluator):
        raise ValueError(f"Evaluator '{path}' is not callable.")
    return evaluator


def load_evaluator_config(config_value: str | None) -> dict[str, Any]:
    if not config_value:
        return {}
    config_path = Path(config_value)
    if config_path.exists():
        return json.loads(config_path.read_text())
    return json.loads(config_value)


def main() -> None:
    args = parse_args()
    evaluator = load_evaluator(args.evaluator)
    evaluator_config = load_evaluator_config(args.evaluator_config)

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
        with override_tm_constants(params):
            metrics = evaluator(params, evaluator_config)
        if "score" not in metrics:
            raise ValueError("Evaluator must return a metrics dict containing 'score'.")
        trials.append(TrialResult(params=params, metrics=metrics))

    trials.sort(key=lambda trial: float(trial.metrics["score"]))
    best_trials = trials[: args.top_k]

    summary = {
        "evaluator": args.evaluator,
        "evaluator_config": evaluator_config,
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
        print(
            json.dumps(
                {
                    "params": trial.params,
                    "metrics": trial.metrics,
                },
                indent=2,
            )
        )

    print(f"\nFull results written to {output_path}")


if __name__ == "__main__":
    main()
