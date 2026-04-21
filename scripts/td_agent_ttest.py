"""Run Welch's independent-samples t-test on TD-agent plateau steps.

This script compares two conditions:
- episodic memory enabled
- episodic memory disabled

It uses `measure_steps` from `src/agents/td_agent.py` to generate samples.

Note: Welch's t-test does not assume equal variances between conditions,
making it more appropriate for RL data where variance may differ greatly.
Verify approximate normality (e.g. via the printed Shapiro-Wilk result)
before interpreting p-values; if the distribution is heavily non-normal,
prefer the Mann-Whitney U test instead.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.td_agent import measure_steps  # noqa: E402


def welch_ttest_fallback(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Welch's two-sample t-test (unequal variances), two-sided.

    Returns (t_statistic, p_value, degrees_of_freedom).
    Falls back to a normal approximation for the p-value when SciPy is absent.
    """
    n1, n2 = len(x), len(y)
    mean1 = sum(x) / n1
    mean2 = sum(y) / n2
    var1 = sum((v - mean1) ** 2 for v in x) / (n1 - 1)
    var2 = sum((v - mean2) ** 2 for v in y) / (n2 - 1)

    se = math.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        return 0.0, 1.0, float(n1 + n2 - 2)

    t = (mean1 - mean2) / se

    # Welch–Satterthwaite degrees of freedom
    dof = (var1 / n1 + var2 / n2) ** 2 / (
        (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    )

    # Normal approximation for p-value (adequate when dof is large)
    p = math.erfc(abs(t) / math.sqrt(2.0))

    return t, p, dof


def shapiro_wilk_note(x: list[float]) -> str:
    """Return a Shapiro-Wilk W statistic and p-value if SciPy is available."""
    try:
        from scipy.stats import shapiro  # type: ignore

        w, p = shapiro(x)
        return f"W={w:.4f}, p={p:.4g}"
    except Exception:
        return "scipy not available"


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


def run_ttest(x: list[int], y: list[int]) -> tuple[float, float, float, str]:
    """Run Welch's t-test with SciPy if available, else fallback implementation."""
    try:
        from scipy.stats import ttest_ind  # type: ignore

        result = ttest_ind(x, y, equal_var=False, alternative="two-sided")
        t, p = float(result.statistic), float(result.pvalue)
        dof = float(result.df)
        return t, p, dof, "scipy"
    except Exception:
        t, p, dof = welch_ttest_fallback([float(v) for v in x], [float(v) for v in y])
        return t, p, dof, "fallback"


def cohens_d(x: list[float], y: list[float]) -> float:
    """Pooled Cohen's d effect size."""
    n1, n2 = len(x), len(y)
    mean1 = np.mean(x)
    mean2 = np.mean(y)
    pooled_sd = math.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return float((mean1 - mean2) / pooled_sd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare plateau steps with/without episodic memory using Welch's t-test."
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

    t_stat, p_value, dof, method = run_ttest(with_memory, without_memory)
    d = cohens_d([float(v) for v in with_memory], [float(v) for v in without_memory])

    print("\nNormality check (Shapiro-Wilk) — t-test assumes approximately normal data")
    print(f"  With episodic memory:    {shapiro_wilk_note([float(v) for v in with_memory])}")
    print(f"  Without episodic memory: {shapiro_wilk_note([float(v) for v in without_memory])}")

    print("\nSummary (Welch's independent-samples t-test)")
    print(f"Runs per condition:     {args.runs}")
    print(f"t-test implementation:  {method}")
    print(f"With episodic memory    -> mean={np.mean(with_memory):.2f}, std={np.std(with_memory, ddof=1):.2f}")
    print(f"Without episodic memory -> mean={np.mean(without_memory):.2f}, std={np.std(without_memory, ddof=1):.2f}")
    print(f"t statistic:            {t_stat:.4f}")
    print(f"Degrees of freedom:     {dof:.2f}")
    print(f"p-value (two-sided):    {p_value:.6g}")
    print(f"Cohen's d:              {d:.4f}")


if __name__ == "__main__":
    main()
