from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.HTM import ColumnField, InputField
from core.brain import Brain
from encoder_layer.rdse import RDSEParameters
from encoder_layer.date_encoder import DateEncoderParameters


@dataclass(frozen=True)
class HotGymConfig:
    num_columns: int = 2048
    cells_per_column: int = 128 
    resolution: float = 0.1
    rdse_seed: int = 5
    train_steps: int = 0
    eval_steps: int = 500
    plot: bool = False
    missing_prediction_penalty: float = 2.0


def parse_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string from CSV format to datetime."""
    return datetime.strptime(ts_str, "%m/%d/%y %H:%M")


def _resolve_config(config: dict[str, Any] | HotGymConfig | None) -> HotGymConfig:
    if config is None:
        return HotGymConfig()
    if isinstance(config, HotGymConfig):
        return config
    return HotGymConfig(**config)


def _build_brain(config: HotGymConfig) -> Brain:
    rdse_params = RDSEParameters(
        size=config.num_columns,
        sparsity=0.02,
        resolution=config.resolution,
        category=False,
        seed=config.rdse_seed,
    )
    consumption_field = InputField(encoder_params=rdse_params)

    date_params = DateEncoderParameters(
        day_of_week_radius=1,
        day_of_week_size=100,
        time_of_day_radius=1,
        time_of_day_size=100,
        weekend_size=20,
    )
    date_field = InputField(encoder_params=date_params)

    l1_column_field = ColumnField(
        input_fields=[consumption_field, date_field],
        non_spatial=True,
        num_columns=config.num_columns,
        cells_per_column=config.cells_per_column,
    )

    l2_column_field = ColumnField(
        input_fields=[l1_column_field],
        non_spatial=True,
        num_columns=config.num_columns,
        cells_per_column=config.cells_per_column,
    )

    l1_column_field.add_input_fields([l2_column_field])

    return Brain(
        {
            "consumption": consumption_field,
            "date": date_field,
            "l1_column_field": l1_column_field,
            "l2_column_field": l2_column_field,
        }
    )


def _load_data() -> pd.DataFrame:
    csv_path = Path(__file__).parent.parent / "data" / "rec-center-hourly.csv"
    df = pd.read_csv(csv_path, skiprows=3, names=["timestamp", "kw_energy_consumption"])
    df["datetime"] = df["timestamp"].apply(parse_timestamp)
    return df


def _train_brain(
    brain: Brain,
    df: pd.DataFrame,
    train_steps: int,
) -> list[int]:
    burst_counts = []
    for idx in tqdm(range(train_steps), desc="Training"):
        brain.step(
            {
                "date": df["datetime"].iloc[idx],
                "consumption": df["kw_energy_consumption"].iloc[idx],
            }
        )
        burst_counts.append(len(brain["l1_column_field"].bursting_columns))
    return burst_counts


def _evaluate_brain(
    brain: Brain,
    df: pd.DataFrame,
    train_steps: int,
    eval_steps: int,
    missing_prediction_penalty: float,
) -> tuple[list[float], list[int], list[float], list[float], int]:
    evaluation_bursts = []
    errors = []
    actual_values = []
    predicted_values = []
    prediction_failures = 0

    for idx in tqdm(range(eval_steps), desc="Evaluation"):
        index = train_steps + idx
        date = df["datetime"].iloc[index]
        value = df["kw_energy_consumption"].iloc[index]

        prediction = brain.prediction()["consumption"]
        brain.step({"date": date, "consumption": value})

        if prediction is None:
            prediction_failures += 1
            errors.append(missing_prediction_penalty)
        else:
            errors.append(abs(value - prediction) ** 2)
        actual_values.append(value)
        predicted_values.append(prediction)
        evaluation_bursts.append(len(brain["l1_column_field"].bursting_columns))

    return (
        errors,
        evaluation_bursts,
        actual_values,
        predicted_values,
        prediction_failures,
    )


def _plot_predictions(actual_values: list[float], predicted_values: list[float]) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(actual_values, label="Actual", alpha=0.8)
    plt.plot(predicted_values, label="Predicted", alpha=0.8)
    plt.xlabel("Time Step")
    plt.ylabel("kW Energy Consumption")
    plt.title("HTM Predictions vs Actual Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/actual_vs_predicted.png", dpi=150)
    plt.show()


def evaluate_hot_gym(
    params: dict[str, float] | None = None,
    config: dict[str, Any] | HotGymConfig | None = None,
) -> dict[str, Any]:
    """Evaluate HTM prediction accuracy on the hot gym dataset."""
    params = params or {}
    config_obj = _resolve_config(config)
    brain = _build_brain(config_obj)
    df = _load_data()

    train_steps = config_obj.train_steps or max(0, len(df) - config_obj.eval_steps)
    eval_steps = min(config_obj.eval_steps, max(0, len(df) - train_steps))

    burst_counts = _train_brain(brain, df, train_steps)
    (
        errors,
        evaluation_bursts,
        actual_values,
        predicted_values,
        prediction_failures,
    ) = _evaluate_brain(
        brain,
        df,
        train_steps,
        eval_steps,
        config_obj.missing_prediction_penalty,
    )

    mae = sum(errors[1:]) / len(errors[1:])
    score = float(mae)

    if config_obj.plot:
        _plot_predictions(actual_values, predicted_values)

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


def main() -> None:
    metrics = evaluate_hot_gym(config={"plot": True})
    print("Mean Absolute Error of predictions:", metrics["mean_abs_error"])
    print("Prediction failures:", metrics["prediction_failures"])
    print("Evaluation bursting columns:", metrics["avg_eval_bursting_columns"])


if __name__ == "__main__":
    main()
