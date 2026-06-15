"""Generate comparison plots for TD-agent plateau analysis.

This script runs two conditions:
- episodic memory enabled
- episodic memory disabled

It produces three plots from the same sampled runs:
- violin + box plot of steps to plateau
- histogram of steps to plateau
- cumulative reward learning curve with 95% confidence bands
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.td_agent_htm import measure_run  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "plots"
COLORS = {
    True: "#0b6e4f",
    False: "#b85c38",
}
LABELS = {
    True: "Episodic memory",
    False: "No episodic memory",
}


def run_condition(runs: int, episodic_memory: bool, max_steps: int, seed_start: int) -> dict[str, object]:
    """Collect step counts and cumulative reward trajectories for one condition."""
    steps: list[int] = []
    cumulative_rewards: list[list[float]] = []
    plateau_count = 0

    for run_idx in tqdm(range(runs), desc=f"episodic_memory={episodic_memory}"):
        seed = seed_start + run_idx
        result = measure_run(
            episodic_memory=episodic_memory,
            max_steps=max_steps,
            seed=seed,
            verbose=False,
        )
        steps_taken = int(result["steps_taken"])
        steps.append(steps_taken)
        cumulative_rewards.append(list(result["cumulative_rewards"]))
        if bool(result["plateau_detected"]):
            plateau_count += 1

    return {
        "steps": steps,
        "cumulative_rewards": cumulative_rewards,
        "plateau_count": plateau_count,
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
    sem = np.divide(
        std,
        np.sqrt(np.maximum(counts, 1)),
        out=np.zeros_like(std),
        where=counts > 0,
    )
    ci = 1.96 * np.nan_to_num(sem)
    return mean, mean - ci, mean + ci


def plot_steps_distribution(results: dict[bool, dict[str, object]], output_dir: Path) -> None:
    """Plot violin and box plot of steps to plateau."""
    fig, ax = plt.subplots(figsize=(8, 6))
    data = [results[True]["steps"], results[False]["steps"]]
    violin = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for body, condition in zip(violin["bodies"], [True, False], strict=True):
        body.set_facecolor(COLORS[condition])
        body.set_edgecolor(COLORS[condition])
        body.set_alpha(0.35)

    box = ax.boxplot(
        data,
        widths=0.22,
        patch_artist=True,
        tick_labels=[LABELS[True], LABELS[False]],
    )
    for patch, condition in zip(box["boxes"], [True, False], strict=True):
        patch.set_facecolor("white")
        patch.set_edgecolor(COLORS[condition])
        patch.set_linewidth(1.8)
    for median in box["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.8)

    ax.set_title("Steps to plateau by condition")
    ax.set_ylabel("Steps")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "td_agent_steps_violin_box.png", dpi=200)
    plt.close(fig)


def plot_steps_histogram(results: dict[bool, dict[str, object]], output_dir: Path) -> None:
    """Plot overlaid histograms of steps to plateau."""
    fig, ax = plt.subplots(figsize=(8, 6))
    step_values = [np.array(results[True]["steps"]), np.array(results[False]["steps"])]
    combined = np.concatenate(step_values) if step_values else np.array([0])
    bins = min(20, max(5, int(np.sqrt(len(combined)))))

    for condition in [True, False]:
        steps = np.array(results[condition]["steps"])
        ax.hist(
            steps,
            bins=bins,
            alpha=0.45,
            color=COLORS[condition],
            label=LABELS[condition],
            edgecolor="white",
        )

    ax.set_title("Histogram of steps to plateau")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "td_agent_steps_histogram.png", dpi=200)
    plt.close(fig)


def plot_cumulative_rewards(
    results: dict[bool, dict[str, object]],
    max_steps: int,
    output_dir: Path,
) -> None:
    """Plot mean cumulative reward with 95% confidence bands."""
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(1, max_steps + 1)

    for condition in [True, False]:
        matrix = cumulative_reward_matrix(results[condition]["cumulative_rewards"], max_steps)
        mean, lower, upper = mean_and_ci(matrix)
        ax.plot(x, mean, color=COLORS[condition], linewidth=2.2, label=LABELS[condition])
        ax.fill_between(x, lower, upper, color=COLORS[condition], alpha=0.18)

    ax.set_title("Cumulative reward over training")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative reward")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "td_agent_cumulative_reward.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TD-agent comparison plots.")
    parser.add_argument("--runs", type=int, default=100, help="Runs per condition.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per run.")
    parser.add_argument("--seed-start", type=int, default=0, help="Initial random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where plot images will be written.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        True: run_condition(
            runs=args.runs,
            episodic_memory=True,
            max_steps=args.max_steps,
            seed_start=args.seed_start,
        ),
        False: run_condition(
            runs=args.runs,
            episodic_memory=False,
            max_steps=args.max_steps,
            seed_start=args.seed_start,
        ),
    }

    plot_steps_distribution(results, args.output_dir)
    plot_steps_histogram(results, args.output_dir)
    plot_cumulative_rewards(results, args.max_steps, args.output_dir)

    print("Generated plots:")
    print(args.output_dir / "td_agent_steps_violin_box.png")
    print(args.output_dir / "td_agent_steps_histogram.png")
    print(args.output_dir / "td_agent_cumulative_reward.png")
    print(
        "Plateau detections: "
        f"with memory={results[True]['plateau_count']}/{args.runs}, "
        f"without memory={results[False]['plateau_count']}/{args.runs}"
    )


if __name__ == "__main__":
    main()
