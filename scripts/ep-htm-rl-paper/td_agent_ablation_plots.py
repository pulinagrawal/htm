"""HTM vs. tabular ablation plots for the TD agent.

Runs the HTM agent (``td_agent_htm.measure_run``) and the no-HTM tabular agent
(``td_agent_tabular.measure_run``) over the same seeds and compares them on:

- cumulative reward over training (mean + 95% CI band)
- steps-until-plateau distribution (violin + box)

Self-contained on purpose: it does not import ``td_agent_plots.py`` (whose
module-level import path is currently broken).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, cast

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parents[1] / "src"
for p in (str(SRC_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.td_agent_htm import measure_run as measure_run_htm  # noqa: E402
from td_agent_tabular import measure_run as measure_run_tabular  # noqa: E402

OUTPUT_DIR = HERE / "plots"

# (label, measure_fn, extra_kwargs, color)
CONDITIONS: list[tuple[str, Callable[..., dict[str, object]], dict[str, object], str]] = [
    ("HTM + episodic memory", measure_run_htm, {"episodic_memory": True}, "#0b6e4f"),
    ("HTM (no episodic)", measure_run_htm, {"episodic_memory": False}, "#3a7ca5"),
    ("Tabular (no HTM)", measure_run_tabular, {}, "#b85c38"),
]


def run_condition(measure_fn, extra_kwargs, runs, max_steps, seed_start):
    steps: list[int] = []
    cumulative_rewards: list[list[float]] = []
    plateau_detected: list[bool] = []
    for run_idx in tqdm(range(runs), desc=str(extra_kwargs) or "tabular"):
        result = measure_fn(
            max_steps=max_steps,
            seed=seed_start + run_idx,
            verbose=False,
            **extra_kwargs,
        )
        steps.append(int(result["steps_taken"]))
        cumulative_rewards.append(list(cast(list[float], result["cumulative_rewards"])))
        plateau_detected.append(bool(result["plateau_detected"]))
    return {
        "steps": steps,
        "cumulative_rewards": cumulative_rewards,
        "plateau_detected": plateau_detected,
        "plateau_count": sum(plateau_detected),
    }


def cumulative_reward_matrix(trajectories: list[list[float]], max_steps: int) -> np.ndarray:
    """Pad cumulative reward trajectories by carrying forward the last value."""
    matrix = np.full((len(trajectories), max_steps), np.nan, dtype=float)
    for idx, trajectory in enumerate(trajectories):
        if not trajectory:
            matrix[idx, :] = 0.0
            continue
        length = min(len(trajectory), max_steps)
        matrix[idx, :length] = trajectory[:length]
        matrix[idx, length:] = trajectory[length - 1]
    return matrix


def mean_and_ci(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean and approximate 95% confidence interval per time step."""
    mean = np.nanmean(matrix, axis=0)
    counts = np.sum(~np.isnan(matrix), axis=0)
    std = np.nanstd(matrix, axis=0, ddof=1)
    sem = np.divide(std, np.sqrt(np.maximum(counts, 1)), out=np.zeros_like(std), where=counts > 0)
    ci = 1.96 * np.nan_to_num(sem)
    return mean, mean - ci, mean + ci


def plot_cumulative_rewards(results, max_steps, output_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(1, max_steps + 1)
    for label, _, _, color in CONDITIONS:
        matrix = cumulative_reward_matrix(results[label]["cumulative_rewards"], max_steps)
        mean, lower, upper = mean_and_ci(matrix)
        ax.plot(x, mean, color=color, linewidth=2.2, label=label)
        ax.fill_between(x, lower, upper, color=color, alpha=0.15)
    ax.set_title("Cumulative reward over training: HTM vs. tabular")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative reward")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = output_dir / "td_agent_ablation_cumulative_reward.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_steps_distribution(results, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [label for label, *_ in CONDITIONS]
    colors = [color for *_, color in CONDITIONS]
    data = [np.asarray(results[label]["steps"], dtype=float) for label in labels]

    violin = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(cast(list[Any], violin["bodies"]), colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)

    box = ax.boxplot(data, widths=0.22, patch_artist=True, tick_labels=labels)
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
        patch.set_linewidth(1.8)
    for median in box["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.8)

    rng = np.random.default_rng(0)
    for idx, (label, color) in enumerate(zip(labels, colors), start=1):
        steps = np.asarray(results[label]["steps"], dtype=float)
        did_plateau = np.asarray(results[label]["plateau_detected"], dtype=bool)
        xvals = np.full(len(steps), idx, dtype=float) + rng.uniform(-0.06, 0.06, size=len(steps))
        ax.scatter(xvals[did_plateau], steps[did_plateau], s=22, alpha=0.5, color=color, edgecolors="none", zorder=3)
        ax.scatter(
            xvals[~did_plateau], steps[~did_plateau], s=38, alpha=0.9,
            facecolors="white", edgecolors=color, linewidths=1.2, zorder=4,
        )

    ax.set_title("Steps until plateau (open markers = censored at max_steps)")
    ax.set_ylabel("Steps")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out = output_dir / "td_agent_ablation_steps.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="HTM vs. tabular TD-agent ablation plots.")
    parser.add_argument("--runs", type=int, default=50, help="Runs per condition.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per run.")
    parser.add_argument("--seed-start", type=int, default=0, help="Initial random seed.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, measure_fn, extra_kwargs, _ in CONDITIONS:
        results[label] = run_condition(
            measure_fn, extra_kwargs, args.runs, args.max_steps, args.seed_start
        )

    reward_png = plot_cumulative_rewards(results, args.max_steps, args.output_dir)
    steps_png = plot_steps_distribution(results, args.output_dir)

    print("\nGenerated plots:")
    print(f"  {reward_png}")
    print(f"  {steps_png}")
    print(f"\nSummary over {args.runs} runs (censored at max_steps={args.max_steps}):")
    for label, *_ in CONDITIONS:
        steps = np.asarray(results[label]["steps"], dtype=float)
        plateau = results[label]["plateau_count"]
        print(
            f"  {label:24s}  steps mean={steps.mean():7.1f}  std={steps.std():6.1f}"
            f"  plateau={plateau}/{args.runs}"
        )


if __name__ == "__main__":
    main()
