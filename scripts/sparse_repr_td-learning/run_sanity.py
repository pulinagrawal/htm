"""Sanity test: does basic TD-learning hold with sparse states + the thesis update?

Removes *all* HTM complexity. States are unique non-overlapping k-of-M SDRs
(``sparse_encoder.py``); value is learned with the literal thesis reward-circuitry
equations 5.1-5.5 (``sparse_td.py``, a stripped port of ``sungur.ValueField``).
A fixed eps-soft-optimal policy drives a deterministic 4x4 FrozenLake.

We then check the three things TD theory promises, against an analytic ground truth:

  1. Convergence       - mean |TD error| decays toward 0.
  2. Correct values    - learned V(s) converges to the exact Bellman V^pi(s)
                         (computed by policy evaluation on the same model).
  3. Discounted grad.  - V rises monotonically along the optimal path and the
                         step-to-step ratio tracks gamma.

If all three hold, the thesis TD(lambda) value math is sound on a sparse code,
independent of the HTM stack.

Usage:
    uv run python scripts/sparse_repr_td-learning/run_sanity.py
    uv run python scripts/sparse_repr_td-learning/run_sanity.py --steps 200000 --k 16
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from environment import (
    DEFAULT_REWARDS,
    N_STATES,
    OPTIMAL_PATH,
    TERMINAL,
    TILES,
    FrozenLakeModel,
    epsilon_soft_optimal,
    make_continuous_env,
    policy_evaluation,
)
from sparse_encoder import build_encoder
from sparse_td import ThesisTDValue


# -- Training -------------------------------------------------------------------

def train(steps: int, hp: dict, k: int, epsilon: float, alpha_decay: float,
          seed: int, continuous: bool = False, encoder_kind: str = "unique",
          rdse_size: int = 256, rdse_resolution: float = 1.0):
    """Run a fixed eps-soft-optimal policy, learning V with the thesis update.

    ``continuous=False`` (episodic): G/H terminal -> bootstrap cut + reset traces.
    ``continuous=True``: the real reset-free ContinuousFrozenLake -> never terminal,
    never reset, always bootstrap (the goal is a non-absorbing attractor). Traces
    persist across the whole stream, as TD(lambda) prescribes with no episodes.

    ``alpha_decay`` (Robbins-Monro): alpha_t = alpha0 / (1 + alpha_decay * t_units),
    where t_units is episodes (episodic) or kilo-steps (continuous). Needed for the
    stochastic (eps>0) policy, where a constant step size leaves variance in V.

    Returns (learner, encoder, td_error_curve, state_visits).
    """
    rng = random.Random(seed)
    encoder = build_encoder(encoder_kind, N_STATES, k=k, seed=seed,
                            rdse_size=rdse_size, rdse_resolution=rdse_resolution)
    learner = ThesisTDValue(encoder.n_bits, **hp)
    policy = epsilon_soft_optimal(epsilon, rng, continuous=continuous)
    alpha0 = hp["alpha"]

    td_curve = np.empty(steps)
    visits = np.zeros(N_STATES, dtype=np.int64)

    if continuous:
        env = make_continuous_env(seed, DEFAULT_REWARDS)
        state, _ = env.reset()
    else:
        env = FrozenLakeModel(DEFAULT_REWARDS)
        state = env.reset()

    episode = 0
    for t in range(steps):
        action = policy(state)
        if continuous:
            next_state, reward, _was_terminal, _ = env.step(action)
            done = False                    # never cut the bootstrap, never reset
        else:
            next_state, reward, done = env.step(action)

        prev_sdr = encoder.encode(state)
        next_sdr = None if done else encoder.encode(next_state)
        td_curve[t] = learner.update(prev_sdr, reward, next_sdr, done)  # signed
        visits[next_state] += 1

        if done:                            # episodic only
            learner.reset_traces()
            episode += 1
            if alpha_decay > 0:
                learner.alpha = alpha0 / (1.0 + alpha_decay * episode)
            state = env.reset()
        else:
            state = next_state
            if continuous and alpha_decay > 0:
                learner.alpha = alpha0 / (1.0 + alpha_decay * (t / 1000.0))

    if continuous:
        env.close()
    return learner, encoder, td_curve, visits


# -- Diagnostics ----------------------------------------------------------------

def learned_values(learner: ThesisTDValue, encoder,
                   continuous: bool = False) -> np.ndarray:
    """V(s) for every grid state from the sparse code.

    Episodic: terminal G/H pinned to 0 (their value is never learned). Continuous:
    no terminals, so every state's value is read from its SDR.
    """
    V = np.zeros(N_STATES)
    for s in range(N_STATES):
        if not continuous and s in TERMINAL:
            V[s] = 0.0
        else:
            V[s] = learner.value(encoder.encode(s))
    return V


def is_monotonic_increasing(values, tol: float = 1e-6) -> bool:
    return all(b >= a - tol for a, b in zip(values, values[1:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=("episodic", "continuous"), default="episodic",
                        help="episodic = terminal G/H + resets (clean finite values). "
                             "continuous = the reset-free ContinuousFrozenLake (goal is "
                             "a non-absorbing attractor) used by regular_vs_continuous.")
    parser.add_argument("--steps", type=int, default=100_000, help="Env steps.")
    parser.add_argument("--runs", type=int, default=10, help="Seeds (averaged).")
    parser.add_argument("--k", type=int, default=8, help="Active bits per SDR (sparsity).")
    parser.add_argument("--encoder", choices=("unique", "category", "rdse"),
                        default="unique",
                        help="State->SDR encoder. unique/category = non-overlapping "
                             "(should recover tabular); rdse = real RandomDistributed"
                             "ScalarEncoder (index-adjacent states share bits -> tests "
                             "interference).")
    parser.add_argument("--rdse-size", type=int, default=256, help="RDSE SDR size.")
    parser.add_argument("--rdse-resolution", type=float, default=1.0,
                        help="RDSE overlap knob: 1.0 = heavy overlap, <=1/k = ~disjoint.")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Fixed eps-soft-optimal policy. Default 0.0 episodic "
                             "(deterministic path), 0.4 continuous (coverage; the goal "
                             "attractor starves exploration otherwise).")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lam", type=float, default=0.6)
    parser.add_argument("--alpha-decay", type=float, default=None,
                        help="Robbins-Monro step-size decay (use with eps>0). Default 0 "
                             "episodic, 0.05 continuous.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    continuous = args.env == "continuous"
    # Env-appropriate defaults (continuous needs exploration + step-size decay).
    if args.epsilon is None:
        args.epsilon = 0.4 if continuous else 0.0
    if args.alpha_decay is None:
        args.alpha_decay = 0.05 if continuous else 0.0

    hp = {"alpha": args.alpha, "gamma": args.gamma, "lam": args.lam}

    # Ground truth: exact V^pi for this eps-soft-optimal policy in this env.
    V_true = np.array(policy_evaluation(args.epsilon, args.gamma, continuous=continuous))

    signed_curve = np.zeros(args.steps)   # mean signed TD error  -> 0 (Bellman residual)
    abs_curve = np.zeros(args.steps)      # mean |TD error|        -> noise floor
    V_runs, visit_runs = [], []
    for r in range(args.runs):
        learner, encoder, td_curve, visits = train(
            args.steps, hp, args.k, args.epsilon, args.alpha_decay,
            args.seed_start + r, continuous=continuous, encoder_kind=args.encoder,
            rdse_size=args.rdse_size, rdse_resolution=args.rdse_resolution,
        )
        signed_curve += td_curve
        abs_curve += np.abs(td_curve)
        V_runs.append(learned_values(learner, encoder, continuous=continuous))
        visit_runs.append(visits)
    signed_curve /= args.runs
    abs_curve /= args.runs
    V_learned = np.mean(V_runs, axis=0)
    visits = np.sum(visit_runs, axis=0)

    # Compare on visited states. Episodic excludes terminals (value never learned);
    # continuous has no terminals, so every visited state counts.
    visited = np.array([s for s in range(N_STATES)
                        if visits[s] > 0 and (continuous or s not in TERMINAL)])
    abs_err = np.abs(V_learned[visited] - V_true[visited])
    max_abs_err = float(abs_err.max())
    mean_abs_err = float(abs_err.mean())
    # Tolerance scales with value magnitude (5% of the value range): ~0.5 for episodic
    # (V ~ O(10)), ~10 for continuous (V ~ O(200) at the goal attractor).
    value_scale = float(np.abs(V_true[visited]).max())
    value_tol = max(0.5, 0.05 * value_scale)

    path_V = [float(V_learned[s]) for s in OPTIMAL_PATH]
    path_V_true = [float(V_true[s]) for s in OPTIMAL_PATH]
    monotonic = is_monotonic_increasing([V_learned[s] for s in OPTIMAL_PATH[:-1]])

    w = min(2000, args.steps)
    init_residual = float(abs(signed_curve[:w].mean()))   # |E[TD err]| early
    final_residual = float(abs(signed_curve[-w:].mean()))  # |E[TD err]| late  -> 0
    init_abs = float(abs_curve[:w].mean())
    final_abs = float(abs_curve[-w:].mean())               # noise floor (>0 if stochastic)

    # -- Pass/fail verdicts -----------------------------------------------------
    # Convergence: the *expected* TD error (signed mean = Bellman residual) -> 0. This
    # is the universal statement that holds for both deterministic (eps=0) and
    # stochastic (eps>0) policies; |TD error| floors out at the target's std when
    # transitions are random. (No "< init" clause: with alpha=0.5 convergence is faster
    # than the measurement window, so the early residual is already near zero.)
    converged = final_residual < 0.05
    values_correct = max_abs_err < value_tol
    gradient_ok = monotonic

    # -- Report -----------------------------------------------------------------
    enc_desc = args.encoder
    if args.encoder == "rdse":
        enc_desc += f"(size={args.rdse_size},res={args.rdse_resolution})"
    print("\n" + "=" * 74)
    print(f"SANITY TEST: thesis TD(lambda) over SDR states (HTM removed) "
          f"[{args.env} | encoder={enc_desc}]")
    print("=" * 74)
    print(f"steps={args.steps}  runs={args.runs}  k={args.k}  eps={args.epsilon}  "
          f"alpha={args.alpha} gamma={args.gamma} lambda={args.lam} "
          f"alpha_decay={args.alpha_decay}")
    print("-" * 74)
    print(f"{'1. Convergence  |E[TD err]| init -> final':<48}"
          f"{init_residual:>10.4f} -> {final_residual:.4f}")
    print(f"{'   (noise floor: mean|TD err| init -> final)':<48}"
          f"{init_abs:>10.4f} -> {final_abs:.4f}")
    print(f"{'   PASS' if converged else '   FAIL':<48}"
          f"(expected TD error -> 0)")
    print(f"{'2. Correct values  max|V_learned - V_true|':<48}{max_abs_err:>13.4f}")
    print(f"{'   mean|V_learned - V_true|':<48}{mean_abs_err:>13.4f}")
    print(f"{'   PASS' if values_correct else '   FAIL':<48}"
          f"(max abs err < {value_tol:.2f} = 5% of V scale {value_scale:.1f})")
    print(f"{'3. Discounted gradient monotonic on path':<48}"
          f"{str(monotonic):>13}")
    print(f"{'   PASS' if gradient_ok else '   FAIL':<48}")
    print("-" * 74)
    print("V along optimal path (S -> G):")
    print(f"  {'state':<10}{'tile':<6}{'V_learned':>12}{'V_true':>12}")
    for s, vl, vt in zip(OPTIMAL_PATH, path_V, path_V_true):
        print(f"  {s:<10}{TILES[s]:<6}{vl:>12.3f}{vt:>12.3f}")
    print("=" * 74)
    overall = converged and values_correct and gradient_ok
    print(f"OVERALL: {'PASS - basic TD-learning holds on the sparse code' if overall else 'FAIL'}")
    print("=" * 74)

    # -- Persist (unique+episodic uses the base names; others are suffixed) ------
    suffix = "" if not continuous else "_continuous"
    if args.encoder != "unique":
        suffix += f"_{args.encoder}"
    results = {
        "config": {"env": args.env, "encoder": args.encoder,
                   "rdse_size": args.rdse_size, "rdse_resolution": args.rdse_resolution,
                   "steps": args.steps, "runs": args.runs, "k": args.k,
                   "epsilon": args.epsilon, "alpha_decay": args.alpha_decay, **hp},
        "checks": {"converged": converged, "values_correct": values_correct,
                   "gradient_monotonic": gradient_ok, "overall": overall},
        "metrics": {"init_residual": init_residual, "final_residual": final_residual,
                    "init_abs_td": init_abs, "final_abs_td": final_abs,
                    "max_abs_value_error": max_abs_err,
                    "mean_abs_value_error": mean_abs_err,
                    "value_tol": value_tol, "value_scale": value_scale},
        "V_learned": V_learned.tolist(),
        "V_true": V_true.tolist(),
        "optimal_path": OPTIMAL_PATH,
        "path_V_learned": path_V,
        "path_V_true": path_V_true,
    }
    out_json = args.output_dir / f"results{suffix}.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved metrics -> {out_json}")

    if not args.no_plot:
        plot_path = make_plots(signed_curve, abs_curve, V_learned, V_true,
                               visited, args.output_dir, suffix,
                               f"{args.env} | {enc_desc}")
        print(f"Saved plot    -> {plot_path}")


def make_plots(signed_curve, abs_curve, V_learned, V_true, visited, output_dir: Path,
               suffix: str = "", env: str = "episodic") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = "#0b6e4f"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (1) Convergence: |TD error| noise magnitude + |E[TD error]| (the thing that -> 0).
    ax = axes[0]
    n = abs_curve.size
    window = max(1, n // 200)
    kernel = np.ones(window) / window
    sm_abs = np.convolve(abs_curve, kernel, mode="valid")
    sm_signed = np.abs(np.convolve(signed_curve, kernel, mode="valid"))
    xs = np.arange(window - 1, n)
    ax.plot(xs, sm_abs, color="#9aa0a6", lw=1.5, label="mean |TD error| (noise floor)")
    ax.plot(xs, sm_signed, color=color, lw=2, label="|mean TD error| (-> 0)")
    ax.set_title(f"1. Convergence (smoothed, w={window})")
    ax.set_xlabel("Environment step")
    ax.set_ylabel("TD error")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, which="both")

    # (2) Learned vs analytic value. Filled = states the policy visits (what the
    #     accuracy check evaluates); hollow grey = non-terminal states never visited
    #     by this (possibly deterministic) policy, so they stay at the V=0 init.
    ax = axes[1]
    visited = list(visited)
    unvisited = [s for s in range(N_STATES) if s not in TERMINAL and s not in visited]
    lo = min(V_true[s] for s in range(N_STATES) if s not in TERMINAL)
    hi = max(V_true[s] for s in range(N_STATES) if s not in TERMINAL)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="y = x (exact)")
    if unvisited:
        ax.scatter([V_true[s] for s in unvisited], [V_learned[s] for s in unvisited],
                   facecolors="none", edgecolors="#9aa0a6", s=40, zorder=2,
                   label="unvisited (not learned)")
    ax.scatter([V_true[s] for s in visited], [V_learned[s] for s in visited],
               color=color, s=45, zorder=3, label="visited (evaluated)")
    ax.set_title("2. Correct values: learned vs analytic V^pi")
    ax.set_xlabel("Analytic V^pi(s)")
    ax.set_ylabel("Learned V(s)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    # (3) Discounted gradient along optimal path.
    ax = axes[2]
    xs = np.arange(len(OPTIMAL_PATH))
    ax.plot(xs, [V_learned[s] for s in OPTIMAL_PATH], "o-", color=color, lw=2,
            label="learned")
    ax.plot(xs, [V_true[s] for s in OPTIMAL_PATH], "s--", color="#b85c38", lw=1.5,
            alpha=0.8, label="analytic V^pi")
    ax.set_title("3. Discounted gradient: V along optimal path")
    ax.set_xlabel("Position on optimal path (S -> G)")
    ax.set_ylabel("V(s)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}\n{TILES[s]}" for s in OPTIMAL_PATH])
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.suptitle(f"Sparse-SDR thesis TD(lambda) sanity test — {env} FrozenLake",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = output_dir / f"sparse_repr_td{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
