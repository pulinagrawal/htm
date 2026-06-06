"""Do basic TD-learning (SARSA) principles hold for ContinuousFrozenLake?

Trains the identical tabular SARSA agent on two environments that differ *only* in
episode handling, then checks the three things TD theory promises:

  1. Policy correctness  - the greedy policy navigates to the goal.
  2. Convergence         - mean |TD error| decays toward 0.
  3. Bellman gradient    - V(s) rises monotonically along the optimal path.

Both learned policies are evaluated in a common episodic test environment so the
"did it learn to reach the goal?" question is comparable across conditions.

Usage:
    uv run python scripts/regular_vs_continuous_fzlake/run_comparison.py
    uv run python scripts/regular_vs_continuous_fzlake/run_comparison.py --runs 30 --steps 20000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from environments import (
    DEFAULT_REWARDS,
    GOAL_STATE,
    OPTIMAL_PATH,
    EpisodicFrozenLake,
    RewardSchedule,
    TILES,
    make_continuous,
)
from td_learning import TabularSARSA

CONDITIONS = ("episodic", "continuous")
COLORS = {"episodic": "#b85c38", "continuous": "#0b6e4f"}
LABELS = {"episodic": "Regular (episodic)", "continuous": "Continuous (no resets)"}


# -- Training loops -------------------------------------------------------------

def run_continuous(reward_schedule: RewardSchedule, n_steps: int, hp: dict, seed: int):
    """Train on ContinuousFrozenLake: never reset, never cut the bootstrap."""
    rng = random.Random(seed)
    env = make_continuous(reward_schedule, seed)
    agent = TabularSARSA(env.action_space.n, rng=rng, **hp)

    state, _ = env.reset()
    action = agent.choose(state)
    td_errors = np.empty(n_steps)
    goal_hits = np.zeros(n_steps, dtype=np.int64)

    for t in range(n_steps):
        next_state, reward, _was_terminal, _ = env.step(action)
        next_action = agent.choose(next_state)
        td_errors[t] = abs(agent.update(state, action, reward, next_state, next_action, done=False))
        goal_hits[t] = int(next_state == GOAL_STATE)
        state, action = next_state, next_action

    env.close()
    return agent, td_errors, goal_hits


def run_episodic(reward_schedule: RewardSchedule, n_steps: int, hp: dict, seed: int):
    """Train on regular gym FrozenLake: terminal bootstrap cutoff + reset on done.

    Budget is matched on *total environment steps* (not episodes) for a fair comparison.
    """
    rng = random.Random(seed)
    env = EpisodicFrozenLake(reward_schedule, seed=seed)
    agent = TabularSARSA(env.action_space.n, rng=rng, **hp)

    state, _ = env.reset()
    action = agent.choose(state)
    td_errors = np.empty(n_steps)
    goal_hits = np.zeros(n_steps, dtype=np.int64)

    for t in range(n_steps):
        next_state, reward, done, _ = env.step(action)
        goal_hits[t] = int(done and next_state == GOAL_STATE)
        if done:
            td_errors[t] = abs(agent.update(state, action, reward, next_state, None, done=True))
            state, _ = env.reset()
            action = agent.choose(state)
        else:
            next_action = agent.choose(next_state)
            td_errors[t] = abs(agent.update(state, action, reward, next_state, next_action, done=False))
            state, action = next_state, next_action

    env.close()
    return agent, td_errors, goal_hits


TRAINERS = {"episodic": run_episodic, "continuous": run_continuous}


# -- Evaluation / diagnostics ---------------------------------------------------

def eval_policy(agent: TabularSARSA, reward_schedule: RewardSchedule,
                episodes: int = 50, max_steps: int = 100) -> tuple[float, float]:
    """Roll out the greedy policy in a common episodic test env.

    Returns (success_rate, mean_steps_to_goal_on_successes).
    """
    env = EpisodicFrozenLake(reward_schedule)
    successes = 0
    lengths: list[int] = []
    for _ in range(episodes):
        state, _ = env.reset()
        for t in range(max_steps):
            state, _, done, _ = env.step(agent.greedy(state))
            if done:
                if state == GOAL_STATE:
                    successes += 1
                    lengths.append(t + 1)
                break
    env.close()
    mean_len = float(np.mean(lengths)) if lengths else float("nan")
    return successes / episodes, mean_len


def path_values(agent: TabularSARSA) -> list[float]:
    return [agent.value(s) for s in OPTIMAL_PATH]


def is_monotonic_increasing(values: list[float], tol: float = 1e-6) -> bool:
    return all(b >= a - tol for a, b in zip(values, values[1:]))


# -- Orchestration --------------------------------------------------------------

def run_condition(condition: str, runs: int, n_steps: int, hp: dict,
                  reward_schedule: RewardSchedule, seed_start: int) -> dict:
    trainer = TRAINERS[condition]
    success_rates, eval_lengths, first_goal, final_td = [], [], [], []
    mono_count = 0
    goal_curve = np.zeros(n_steps)          # mean cumulative goal hits over time
    td_curve = np.zeros(n_steps)            # mean |TD error| over time
    path_value_runs: list[list[float]] = []

    for run_idx in tqdm(range(runs), desc=condition):
        seed = seed_start + run_idx
        agent, td_errors, goal_hits = trainer(reward_schedule, n_steps, hp, seed)

        sr, mlen = eval_policy(agent, reward_schedule)
        success_rates.append(sr)
        eval_lengths.append(mlen)

        hit_idx = np.flatnonzero(goal_hits)
        first_goal.append(int(hit_idx[0]) if hit_idx.size else n_steps)
        final_td.append(float(np.mean(td_errors[-min(1000, n_steps):])))

        pv = path_values(agent)
        path_value_runs.append(pv)
        mono_count += int(is_monotonic_increasing(pv))

        goal_curve += np.cumsum(goal_hits)
        td_curve += td_errors

    goal_curve /= runs
    td_curve /= runs

    return {
        "condition": condition,
        "success_rate": success_rates,
        "eval_len": eval_lengths,
        "first_goal": first_goal,
        "final_td": final_td,
        "monotonic_frac": mono_count / runs,
        "goal_curve": goal_curve,
        "td_curve": td_curve,
        "path_values_mean": np.nanmean(np.array(path_value_runs), axis=0).tolist(),
    }


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def make_plots(results: dict[str, dict], n_steps: int, output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(n_steps)

    # (1) Learning curve: cumulative goal visits over training steps.
    ax = axes[0]
    for cond in CONDITIONS:
        ax.plot(x, results[cond]["goal_curve"], color=COLORS[cond], lw=2, label=LABELS[cond])
    ax.set_title("Learning: cumulative goal visits")
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Mean cumulative goal visits")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    # (2) Convergence: smoothed mean |TD error| -> should decay toward 0.
    ax = axes[1]
    window = max(1, n_steps // 200)
    xs = np.arange(window - 1, n_steps)
    for cond in CONDITIONS:
        ax.plot(xs, smooth(results[cond]["td_curve"], window), color=COLORS[cond], lw=2, label=LABELS[cond])
    ax.set_title(f"Convergence: mean |TD error| (smoothed, w={window})")
    ax.set_xlabel("Environment step")
    ax.set_ylabel("|TD error|")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, which="both")

    # (3) Bellman value gradient along the optimal path.
    ax = axes[2]
    xs = np.arange(len(OPTIMAL_PATH))
    for cond in CONDITIONS:
        ax.plot(xs, results[cond]["path_values_mean"], "o-", color=COLORS[cond], lw=2, label=LABELS[cond])
    ax.set_title("Bellman gradient: V along optimal path")
    ax.set_xlabel("Position on optimal path (S -> G)")
    ax.set_ylabel("V(s) = max_a Q(s,a)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}\n{TILES[s]}" for s in OPTIMAL_PATH])
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    out = output_dir / "regular_vs_continuous_td.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def print_summary(results: dict[str, dict]) -> None:
    def agg(cond, key):
        # nanmean: runs that never reached the goal contribute nan eval-length.
        return float(np.nanmean(results[cond][key]))

    print("\n" + "=" * 78)
    print("TD-LEARNING: REGULAR vs CONTINUOUS FROZENLAKE")
    print("=" * 78)
    header = f"{'metric':<42}{'episodic':>16}{'continuous':>16}"
    print(header)
    print("-" * 78)
    rows = [
        ("Greedy success rate (-> goal)", "success_rate", "{:.1%}"),
        ("Mean steps-to-goal at eval", "eval_len", "{:.2f}"),
        ("Steps until first goal visit", "first_goal", "{:.0f}"),
        ("Final mean |TD error| (last 1k)", "final_td", "{:.4f}"),
    ]
    for label, key, fmt in rows:
        e = fmt.format(agg("episodic", key))
        c = fmt.format(agg("continuous", key))
        print(f"{label:<42}{e:>16}{c:>16}")
    me = "{:.1%}".format(results["episodic"]["monotonic_frac"])
    mc = "{:.1%}".format(results["continuous"]["monotonic_frac"])
    print(f"{'V monotonic along optimal path':<42}{me:>16}{mc:>16}")
    print("-" * 78)
    print(f"{'V(start)':<42}{results['episodic']['path_values_mean'][0]:>16.2f}"
          f"{results['continuous']['path_values_mean'][0]:>16.2f}")
    print(f"{'V(goal)':<42}{results['episodic']['path_values_mean'][-1]:>16.2f}"
          f"{results['continuous']['path_values_mean'][-1]:>16.2f}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30, help="Seeds per condition.")
    parser.add_argument("--steps", type=int, default=20000, help="Env steps per run.")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    hp = {"alpha": args.alpha, "gamma": args.gamma, "epsilon": args.epsilon}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        cond: run_condition(cond, args.runs, args.steps, hp, DEFAULT_REWARDS, args.seed_start)
        for cond in CONDITIONS
    }

    print_summary(results)

    # Persist scalar metrics (drop the big curves to keep the JSON readable).
    serializable = {
        cond: {k: v for k, v in r.items() if k not in ("goal_curve", "td_curve")}
        for cond, r in results.items()
    }
    json_path = args.output_dir / "results.json"
    json_path.write_text(json.dumps(serializable, indent=2))
    print(f"\nSaved metrics -> {json_path}")

    if not args.no_plot:
        plot_path = make_plots(results, args.steps, args.output_dir)
        print(f"Saved plot    -> {plot_path}")


if __name__ == "__main__":
    main()
