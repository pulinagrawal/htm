"""Run Mann-Whitney U analysis on TD-agent plateau steps.

This script compares two conditions:
- episodic memory enabled
- episodic memory disabled

It uses `measure_steps` from `src/agents/td_agent.py` to generate samples.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.td_agent_htm import measure_steps  # noqa: E402


def _average_ranks(values: list[tuple[float, int]]) -> list[float]:
    """Return average rank for each sorted item (1-indexed rank values)."""
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1][0] == values[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    return ranks


def mann_whitney_u_fallback(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    """Compute Mann-Whitney U and two-sided p-value via normal approximation.

    Includes tie correction and continuity correction.
    """
    x_vals = [float(v) for v in x]
    y_vals = [float(v) for v in y]
    n1 = len(x_vals)
    n2 = len(y_vals)
    if n1 == 0 or n2 == 0:
        raise ValueError("Both samples must be non-empty.")

    combined = [(v, 0) for v in x_vals] + [(v, 1) for v in y_vals]
    combined.sort(key=lambda item: item[0])
    ranks = _average_ranks(combined)

    rank_sum_x = sum(rank for rank, (_, group) in zip(ranks, combined) if group == 0)
    u1 = rank_sum_x - (n1 * (n1 + 1) / 2.0)
    u2 = n1 * n2 - u1
    u_stat = min(u1, u2)

    n = n1 + n2
    tie_counts: dict[float, int] = {}
    for value, _ in combined:
        tie_counts[value] = tie_counts.get(value, 0) + 1
    tie_term = sum((t**3 - t) for t in tie_counts.values())

    if n <= 1:
        return u_stat, 1.0

    variance = (n1 * n2 / 12.0) * ((n + 1) - (tie_term / (n * (n - 1))))
    if variance <= 0:
        return u_stat, 1.0

    mean_u = n1 * n2 / 2.0
    z = (abs(u_stat - mean_u) - 0.5) / math.sqrt(variance)
    p_two_sided = math.erfc(abs(z) / math.sqrt(2.0))
    return u_stat, p_two_sided


def run_condition(runs: int, episodic_memory: bool, max_steps: int, seed_start: int) -> list[int]:
    """Run repeated simulations for one condition and return steps-to-plateau."""
    results: list[int] = []
    for run_idx in tqdm(range(runs), desc=f"episodic_memory={episodic_memory}"):
        seed = seed_start + run_idx
        steps = measure_steps(
            episodic_memory=episodic_memory,
            max_steps=max_steps,
            seed=seed,
            verbose=False,
        )
        results.append(int(steps))
    return results


def run_mann_whitney(x: list[int], y: list[int]) -> tuple[float, float, str]:
    """Run Mann-Whitney U with SciPy if available, else fallback implementation."""
    try:
        from scipy.stats import mannwhitneyu  # type: ignore

        result = mannwhitneyu(x, y, alternative="two-sided")
        return float(result.statistic), float(result.pvalue), "scipy"
    except Exception:
        u_stat, p_value = mann_whitney_u_fallback(x, y)
        return float(u_stat), float(p_value), "fallback"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare plateau steps with and without episodic memory using Mann-Whitney U."
    )
    parser.add_argument("--runs", type=int, default=100, help="Runs per condition.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per run.")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Initial seed used for reproducible runs.",
    )
    args = parser.parse_args()

    with_memory = run_condition(
        runs=args.runs,
        episodic_memory=True,
        max_steps=args.max_steps,
        seed_start=args.seed_start,
    )
    without_memory = run_condition(
        runs=args.runs,
        episodic_memory=False,
        max_steps=args.max_steps,
        seed_start=args.seed_start,
    )

    u_stat, p_value, method = run_mann_whitney(with_memory, without_memory)

    n1 = len(with_memory)
    n2 = len(without_memory)
    u1 = float(sum(1 for a in with_memory for b in without_memory if a > b) + 0.5 * sum(1 for a in with_memory for b in without_memory if a == b))
    common_language = u1 / (n1 * n2)

    print("\nSummary")
    print(f"Runs per condition: {args.runs}")
    print(f"Mann-Whitney implementation: {method}")
    print(f"With episodic memory  -> mean={np.mean(with_memory):.2f}, median={np.median(with_memory):.2f}")
    print(f"Without episodic mem. -> mean={np.mean(without_memory):.2f}, median={np.median(without_memory):.2f}")
    print(f"U statistic: {u_stat:.3f}")
    print(f"p-value (two-sided): {p_value:.6g}")
    print(f"Common-language effect size P(X>Y): {common_language:.3f}")


if __name__ == "__main__":
    main()
