"""Can the thesis control logic (D1/D2 Go/No-Go + voluntary action), built on the sparse
TD value learner, actually *solve* episodic FrozenLake?

This is the control counterpart to ``run_sanity.py`` (which only checked the value math).
Here the agent must choose actions and navigate S→G. Architecture: ``GoNoGoActor``
(``sparse_control.py``) over a sparse SDR encoding, learning its one-step model online.

We report, over training:
  - learning curve   : per-episode success (reached goal) and return, rolling mean;
  - greedy success   : roll out the greedy (target) policy in a fresh env — the headline
                       "did it solve it?" number;
  - value gradient   : learned V = V_go − V_nogo along the optimal path (should rise);
  - Go/No-Go split   : confirms D1 ≠ D2 (the B7 fix), so the motor signal is nonzero.

Usage:
    uv run python scripts/sparse_repr_td-learning/run_control.py
    uv run python scripts/sparse_repr_td-learning/run_control.py --episodes 4000 --runs 10
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from environment import (
    GOAL_STATE,
    N_ACTIONS,
    N_STATES,
    OPTIMAL_PATH,
    TILES,
    FrozenLakeModel,
    make_continuous_env,
)
from sparse_control import GoNoGoActor
from sparse_encoder import build_encoder


def run_once(episodes: int, max_steps: int, hp: dict, base_excitation: float,
             encoder_kind: str, k: int, seed: int, epsilon: float = 0.0):
    """Train one agent on episodic FrozenLake. Returns (agent, encoder, curves)."""
    rng = np.random.default_rng(seed)
    encoder = build_encoder(encoder_kind, N_STATES, k=k, seed=seed)
    env = FrozenLakeModel()
    agent = GoNoGoActor(encoder.n_bits, N_ACTIONS, base_excitation=base_excitation,
                        epsilon=epsilon, rng=rng, **hp)

    success = np.zeros(episodes)
    returns = np.zeros(episodes)
    for ep in range(episodes):
        state = env.reset()
        sdr = encoder.encode(state)
        for _ in range(max_steps):
            action = agent.act(sdr)
            next_state, reward, done = env.step(action)
            next_sdr = encoder.encode(next_state)
            agent.observe(sdr, action, reward, next_sdr, done)
            returns[ep] += reward
            if done:
                success[ep] = float(next_state == GOAL_STATE)
                break
            state, sdr = next_state, next_sdr
        agent.end_episode()

    return agent, encoder, success, returns


def run_continuous(steps: int, hp: dict, base_excitation: float, encoder_kind: str,
                   k: int, seed: int, eval_every: int, epsilon: float = 0.0):
    """Train on the reset-free ContinuousFrozenLake: one long stream, no episodes,
    no terminal bootstrap cut (the goal is a non-absorbing attractor).

    Returns (agent, encoder, goal_hit_curve, eval_steps, eval_success).
    """
    rng = np.random.default_rng(seed)
    encoder = build_encoder(encoder_kind, N_STATES, k=k, seed=seed)
    env = make_continuous_env(seed)
    agent = GoNoGoActor(encoder.n_bits, N_ACTIONS, base_excitation=base_excitation,
                        epsilon=epsilon, rng=rng, **hp)

    goal_hits = np.zeros(steps)
    eval_steps: list[int] = []
    eval_success: list[float] = []

    state, _ = env.reset()
    sdr = encoder.encode(state)
    eps0, eps_min = epsilon, min(0.05, epsilon)
    for t in range(steps):
        # GLIE: anneal exploration linearly eps0 -> eps_min so the on-policy value
        # converges toward the greedy policy (a fixed high eps leaves the critic tracking
        # a half-random policy, whose greedy argmax flips and breaks the route).
        agent.epsilon = eps0 + (eps_min - eps0) * (t / steps)
        action = agent.act(sdr)
        next_state, reward, _term, _ = env.step(action)   # term ignored: never resets
        next_sdr = encoder.encode(next_state)
        agent.observe(sdr, action, reward, next_sdr, done=False)
        goal_hits[t] = float(next_state == GOAL_STATE)
        state, sdr = next_state, next_sdr
        if eval_every and (t + 1) % eval_every == 0:
            sr, _ = eval_greedy(agent, encoder)
            eval_steps.append(t + 1)
            eval_success.append(sr)

    env.close()
    return agent, encoder, goal_hits, eval_steps, eval_success


def eval_greedy(agent: GoNoGoActor, encoder, episodes: int = 200,
                max_steps: int = 100) -> tuple[float, float]:
    """Roll out the greedy target policy in a fresh env. Returns (success, mean_len)."""
    env = FrozenLakeModel()
    wins, lengths = 0, []
    for _ in range(episodes):
        state = env.reset()
        for t in range(max_steps):
            state, _r, done = env.step(agent.greedy(encoder.encode(state)))
            if done:
                if state == GOAL_STATE:
                    wins += 1
                    lengths.append(t + 1)
                break
    return wins / episodes, float(np.mean(lengths)) if lengths else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=("episodic", "continuous"), default="episodic",
                        help="episodic = terminal G/H + resets. continuous = reset-free "
                             "ContinuousFrozenLake (thesis default; goal is a "
                             "non-absorbing attractor).")
    parser.add_argument("--episodes", type=int, default=3000,
                        help="Episodic only: number of episodes.")
    parser.add_argument("--steps", type=int, default=400_000,
                        help="Continuous only: length of the single training stream.")
    parser.add_argument("--eval-every", type=int, default=20_000,
                        help="Continuous only: greedy-eval checkpoint interval (steps).")
    parser.add_argument("--runs", type=int, default=8, help="Seeds (averaged).")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--encoder", choices=("unique", "category", "rdse"),
                        default="unique")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lam", type=float, default=0.6)
    parser.add_argument("--base-excitation", type=float, default=4.0,
                        help="Motor base excitation magnitude (WTA tie-break / local "
                             "exploration).")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Explicit random-action rate. Default 0.0 episodic, 0.6 "
                             "continuous (annealed -> 0.05 over the run). The reset-free "
                             "goal-attractor needs a high, GLIE-annealed exploration rate, "
                             "as regular_vs_continuous found (~0.6).")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.epsilon is None:
        args.epsilon = 0.6 if args.env == "continuous" else 0.0

    hp = {"alpha": args.alpha, "gamma": args.gamma, "lam": args.lam}

    if args.env == "continuous":
        main_continuous(args, hp)
        return

    success_curves = np.zeros(args.episodes)
    return_curves = np.zeros(args.episodes)
    greedy_success, greedy_len = [], []
    path_V_runs, gonogo_gap = [], []

    for r in tqdm(range(args.runs), desc="control"):
        agent, encoder, success, returns = run_once(
            args.episodes, args.max_steps, hp, args.base_excitation,
            args.encoder, args.k, args.seed_start + r, epsilon=args.epsilon)
        success_curves += success
        return_curves += returns

        sr, mlen = eval_greedy(agent, encoder)
        greedy_success.append(sr)
        greedy_len.append(mlen)

        path_V_runs.append([agent.value(encoder.encode(s)) for s in OPTIMAL_PATH])
        # Go≠NoGo check: mean |V_go − V_nogo|... no, we want they are NOT identical.
        go = np.array([agent.value_go(encoder.encode(s)) for s in range(N_STATES)])
        ng = np.array([agent.value_nogo(encoder.encode(s)) for s in range(N_STATES)])
        gonogo_gap.append(float(np.abs(go - ng).mean()))

    success_curves /= args.runs
    return_curves /= args.runs
    path_V = np.mean(path_V_runs, axis=0).tolist()
    mean_greedy = float(np.mean(greedy_success))
    std_greedy = float(np.std(greedy_success))
    mean_len = float(np.nanmean(greedy_len))

    solved = mean_greedy >= 0.95

    # -- report -----------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"THESIS CONTROL (D1/D2 Go/No-Go + voluntary action) on episodic FrozenLake "
          f"[encoder={args.encoder}]")
    print("=" * 74)
    print(f"episodes={args.episodes}  runs={args.runs}  base_excitation={args.base_excitation}"
          f"  alpha={args.alpha} gamma={args.gamma} lambda={args.lam}")
    print("-" * 74)
    print(f"{'Greedy success rate (-> goal)':<40}{mean_greedy:>10.1%}  (+/- {std_greedy:.1%})")
    print(f"{'Mean greedy steps-to-goal':<40}{mean_len:>10.2f}  (optimal = 6)")
    print(f"{'Final-100-ep training success':<40}{success_curves[-100:].mean():>10.1%}")
    print(f"{'Go vs No-Go mean |V_go - V_nogo|':<40}{np.mean(gonogo_gap):>10.3f}  "
          f"(B7: must be > 0)")
    print("-" * 74)
    print("Learned V = V_go - V_nogo along optimal path (S -> G):")
    print(f"  {'state':<8}{'tile':<6}{'V':>10}")
    for s, v in zip(OPTIMAL_PATH, path_V):
        print(f"  {s:<8}{TILES[s]:<6}{v:>10.3f}")
    print("=" * 74)
    print(f"SOLVED: {'YES — greedy policy reaches the goal' if solved else 'NO'} "
          f"(threshold 95%)")
    print("=" * 74)

    results = {
        "config": {"episodes": args.episodes, "runs": args.runs,
                   "base_excitation": args.base_excitation, "encoder": args.encoder, **hp},
        "solved": solved,
        "greedy_success_mean": mean_greedy, "greedy_success_std": std_greedy,
        "greedy_steps": mean_len,
        "gonogo_gap_mean": float(np.mean(gonogo_gap)),
        "path_V": path_V, "optimal_path": OPTIMAL_PATH,
    }
    suffix = "" if args.encoder == "unique" else f"_{args.encoder}"
    out_json = args.output_dir / f"results_control{suffix}.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved metrics -> {out_json}")

    if not args.no_plot:
        plot_path = make_plots(success_curves, return_curves, path_V,
                               args.output_dir, suffix, args.encoder)
        print(f"Saved plot    -> {plot_path}")


def main_continuous(args, hp: dict) -> None:
    """Control on the reset-free ContinuousFrozenLake (thesis default)."""
    goal_curve = np.zeros(args.steps)
    eval_success_runs: list[list[float]] = []
    eval_steps: list[int] = []
    greedy_success, greedy_len, path_V_runs, gonogo_gap = [], [], [], []

    for r in tqdm(range(args.runs), desc="control(cont)"):
        agent, encoder, goal_hits, esteps, esucc = run_continuous(
            args.steps, hp, args.base_excitation, args.encoder, args.k,
            args.seed_start + r, args.eval_every, epsilon=args.epsilon)
        goal_curve += goal_hits
        eval_steps = esteps
        eval_success_runs.append(esucc)

        sr, mlen = eval_greedy(agent, encoder)
        greedy_success.append(sr)
        greedy_len.append(mlen)
        path_V_runs.append([agent.value(encoder.encode(s)) for s in OPTIMAL_PATH])
        go = np.array([agent.value_go(encoder.encode(s)) for s in range(N_STATES)])
        ng = np.array([agent.value_nogo(encoder.encode(s)) for s in range(N_STATES)])
        gonogo_gap.append(float(np.abs(go - ng).mean()))

    goal_curve /= args.runs
    eval_curve = np.mean(eval_success_runs, axis=0) if eval_steps else np.array([])
    path_V = np.mean(path_V_runs, axis=0).tolist()
    mean_greedy = float(np.mean(greedy_success))
    std_greedy = float(np.std(greedy_success))
    mean_len = float(np.nanmean(greedy_len))
    solved = mean_greedy >= 0.95

    print("\n" + "=" * 74)
    print(f"THESIS CONTROL (D1/D2 Go/No-Go + voluntary action) on CONTINUOUS FrozenLake "
          f"[encoder={args.encoder}]")
    print("=" * 74)
    print(f"steps={args.steps}  runs={args.runs}  epsilon={args.epsilon}  "
          f"base_excitation={args.base_excitation}"
          f"  alpha={args.alpha} gamma={args.gamma} lambda={args.lam}")
    print("-" * 74)
    print(f"{'Greedy success rate (-> goal from S)':<40}{mean_greedy:>10.1%}  "
          f"(+/- {std_greedy:.1%})")
    print(f"{'Mean greedy steps-to-goal':<40}{mean_len:>10.2f}  (optimal = 6)")
    print(f"{'Go vs No-Go mean |V_go - V_nogo|':<40}{np.mean(gonogo_gap):>10.3f}  "
          f"(B7: must be > 0)")
    if eval_steps:
        ckpts = ", ".join(f"{s//1000}k:{v:.0%}" for s, v in zip(eval_steps, eval_curve))
        print(f"Greedy success over training: {ckpts}")
    print("-" * 74)
    print("Learned V = V_go - V_nogo along optimal path (S -> G):")
    print(f"  {'state':<8}{'tile':<6}{'V':>10}")
    for s, v in zip(OPTIMAL_PATH, path_V):
        print(f"  {s:<8}{TILES[s]:<6}{v:>10.3f}")
    print("=" * 74)
    print(f"SOLVED: {'YES — greedy policy reaches the goal from S' if solved else 'NO'} "
          f"(threshold 95%)")
    print("=" * 74)

    results = {
        "config": {"env": "continuous", "steps": args.steps, "runs": args.runs,
                   "base_excitation": args.base_excitation, "encoder": args.encoder, **hp},
        "solved": solved, "greedy_success_mean": mean_greedy,
        "greedy_success_std": std_greedy, "greedy_steps": mean_len,
        "gonogo_gap_mean": float(np.mean(gonogo_gap)),
        "eval_steps": eval_steps, "eval_success": eval_curve.tolist() if eval_steps else [],
        "path_V": path_V, "optimal_path": OPTIMAL_PATH,
    }
    suffix = "_continuous" + ("" if args.encoder == "unique" else f"_{args.encoder}")
    out_json = args.output_dir / f"results_control{suffix}.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved metrics -> {out_json}")

    if not args.no_plot:
        plot_path = make_plots_continuous(goal_curve, eval_steps, eval_curve, path_V,
                                          args.output_dir, suffix, args.encoder)
        print(f"Saved plot    -> {plot_path}")


def make_plots_continuous(goal_curve, eval_steps, eval_curve, path_V, output_dir: Path,
                          suffix: str, encoder: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = "#0b6e4f"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    n = goal_curve.size
    w = max(1, n // 200)

    ax = axes[0]
    ax.plot(np.arange(w - 1, n), smooth(goal_curve, w), color=color, lw=2)
    ax.set_title(f"Behaviour: goal-visit rate per step (rolling, w={w})")
    ax.set_xlabel("Environment step")
    ax.set_ylabel("P(on goal)")
    ax.grid(alpha=0.2)

    ax = axes[1]
    if eval_steps:
        ax.plot(eval_steps, eval_curve, "o-", color="#b85c38", lw=2)
    ax.set_title("Greedy success from S over training")
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Greedy success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)

    ax = axes[2]
    xs = np.arange(len(OPTIMAL_PATH))
    ax.plot(xs, path_V, "o-", color=color, lw=2)
    ax.set_title("Learned V = V_go - V_nogo along optimal path")
    ax.set_xlabel("Position on optimal path (S -> G)")
    ax.set_ylabel("V(s)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}\n{TILES[s]}" for s in OPTIMAL_PATH])
    ax.grid(alpha=0.2)

    fig.suptitle(f"Thesis control on CONTINUOUS FrozenLake — encoder={encoder}",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = output_dir / f"sparse_control{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def make_plots(success_curve, return_curve, path_V, output_dir: Path,
               suffix: str, encoder: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = "#0b6e4f"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    n = success_curve.size
    w = max(1, n // 100)

    ax = axes[0]
    ax.plot(np.arange(w - 1, n), smooth(success_curve, w), color=color, lw=2)
    ax.set_title(f"Learning: episode success rate (rolling, w={w})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("P(reached goal)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.plot(np.arange(w - 1, n), smooth(return_curve, w), color="#b85c38", lw=2)
    ax.set_title(f"Learning: episode return (rolling, w={w})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Sum of rewards")
    ax.grid(alpha=0.2)

    ax = axes[2]
    xs = np.arange(len(OPTIMAL_PATH))
    ax.plot(xs, path_V, "o-", color=color, lw=2)
    ax.set_title("Learned V = V_go - V_nogo along optimal path")
    ax.set_xlabel("Position on optimal path (S -> G)")
    ax.set_ylabel("V(s)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}\n{TILES[s]}" for s in OPTIMAL_PATH])
    ax.grid(alpha=0.2)

    fig.suptitle(f"Thesis control on episodic FrozenLake — encoder={encoder}",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = output_dir / f"sparse_control{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
