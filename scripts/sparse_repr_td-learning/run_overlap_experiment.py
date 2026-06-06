"""Does *grid-distance-aware* overlap help (or hurt) the thesis TD value learner?

The 4x4 sanity test (``run_sanity.py``) showed the thesis TD(lambda) update is exact
on **non-overlapping** codes (``unique``/``category``) and *biased* when overlap is
**misaligned** with value — the ``rdse`` adapter encodes the *state index as a scalar*,
sharing bits between index-adjacent (but value-distinct) tiles.

This experiment asks the complementary question the README left open: *what if the
overlap is **spatially** meaningful?* We

  1. enlarge the env to the standard **8x8** FrozenLake (64 states) so there is real
     room between tiles for a distance gradient to matter (``grid_env.py``);
  2. introduce ``CoordinateEncoder`` — encode each tile's **(row, col)** with a real
     project encoder per axis and concatenate, so code overlap falls off with **grid
     (diagonal) distance** (``coordinate_encoder.py``); and
  3. run it with **both backends** the user asked for — a contiguous ``scalar`` encoder
     and a ``rdse`` encoder — at **1024-bit** dimensionality.

For contrast we also run the non-overlapping ``unique`` baseline and the old
value-misaligned ``index-rdse`` (state index as a scalar, the 4x4 FAIL case) at the
same 1024 bits. Every encoder is scored on the same three TD checks against the exact
analytic V^pi, plus an **overlap-vs-distance** diagnostic.

Usage:
    uv run python scripts/sparse_repr_td-learning/run_overlap_experiment.py
    uv run python scripts/sparse_repr_td-learning/run_overlap_experiment.py \
        --encoders coord-scalar coord-rdse --steps 300000 --runs 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
from pathlib import Path

import numpy as np

# Silence the project encoders' per-encode INFO logging.
logging.disable(logging.INFO)

from coordinate_encoder import CoordinateEncoder
from grid_env import DEFAULT_REWARDS, GridWorld, epsilon_soft_optimal, policy_evaluation
from sparse_encoder import RDSEEncoderAdapter, UniqueSDREncoder
from sparse_td import ThesisTDValue

ENCODERS = ("unique", "index-rdse", "coord-scalar", "coord-rdse")


# -- Encoder factory ------------------------------------------------------------

def build_encoder(kind: str, env: GridWorld, size: int, radius_tiles: float,
                  seed: int):
    """Return an object with ``.encode(state)->tuple`` and ``.n_bits``/``.k``."""
    n = env.n_states
    if kind == "unique":
        # Disjoint k-of-M; k=size/n gives ~size bits, fully non-overlapping baseline.
        enc = UniqueSDREncoder(n, k=max(1, size // n), seed=seed)
        enc.k = max(1, size // n)
        return enc
    if kind == "index-rdse":
        # The 4x4 FAIL case at 1024 bits: state *index* as a scalar (overlap is
        # index-adjacency, which is NOT grid/value-adjacency).
        enc = RDSEEncoderAdapter(n, k=24, size=size, resolution=1.0, seed=seed)
        enc.k = 24
        return enc
    if kind == "coord-scalar":
        return CoordinateEncoder(env.rows, env.cols, backend="scalar", size=size,
                                 radius_tiles=radius_tiles, seed=seed)
    if kind == "coord-rdse":
        return CoordinateEncoder(env.rows, env.cols, backend="rdse", size=size,
                                 radius_tiles=radius_tiles, seed=seed)
    raise ValueError(f"unknown encoder: {kind}")


# -- Overlap-vs-distance diagnostic --------------------------------------------

def overlap_distance_stats(encoder, env: GridWorld) -> dict:
    """Correlate code overlap with grid (Chebyshev/Euclidean) distance over all pairs."""
    codes = {s: set(encoder.encode(s)) for s in range(env.n_states)}
    cheb, eucl, ov = [], [], []
    for a, b in itertools.combinations(range(env.n_states), 2):
        ra, ca = env.coords(a)
        rb, cb = env.coords(b)
        cheb.append(max(abs(ra - rb), abs(ca - cb)))
        eucl.append(((ra - rb) ** 2 + (ca - cb) ** 2) ** 0.5)
        ov.append(len(codes[a] & codes[b]))
    cheb, eucl, ov = np.array(cheb), np.array(eucl), np.array(ov)
    by_cheb = {int(d): float(ov[cheb == d].mean()) for d in np.unique(cheb)}
    # A non-overlapping code has zero variance -> undefined correlation; report 0.
    corr = lambda d: 0.0 if ov.std() == 0 else float(np.corrcoef(ov, d)[0, 1])
    return {
        "corr_overlap_chebyshev": corr(cheb),
        "corr_overlap_euclidean": corr(eucl),
        "mean_overlap_by_chebyshev": by_cheb,
        "max_pair_overlap": int(ov.max()),
    }


# -- Training -------------------------------------------------------------------

def train(env: GridWorld, encoder, hp: dict, steps: int, epsilon: float,
          alpha_decay: float, gamma: float, seed: int):
    """Fixed eps-soft-optimal policy on episodic GridWorld, thesis TD(lambda) value."""
    rng = random.Random(seed)
    learner = ThesisTDValue(encoder.n_bits, **hp)
    policy = epsilon_soft_optimal(env, epsilon, gamma, rng)
    alpha0 = hp["alpha"]

    td_curve = np.empty(steps)
    visits = np.zeros(env.n_states, dtype=np.int64)
    state = env.reset()
    episode = 0
    for t in range(steps):
        action = policy(state)
        next_state, reward, done = env.step(action)
        prev_sdr = encoder.encode(state)
        next_sdr = None if done else encoder.encode(next_state)
        td_curve[t] = learner.update(prev_sdr, reward, next_sdr, done)
        visits[next_state] += 1
        if done:
            learner.reset_traces()
            episode += 1
            if alpha_decay > 0:
                learner.alpha = alpha0 / (1.0 + alpha_decay * episode)
            state = env.reset()
        else:
            state = next_state
    return learner, td_curve, visits


def learned_values(learner, encoder, env: GridWorld) -> np.ndarray:
    V = np.zeros(env.n_states)
    for s in range(env.n_states):
        V[s] = 0.0 if s in env.terminal else learner.value(encoder.encode(s))
    return V


def is_monotonic(values, tol: float = 1e-6) -> bool:
    return all(b >= a - tol for a, b in zip(values, values[1:]))


# -- One encoder's run ----------------------------------------------------------

def run_encoder(kind: str, env: GridWorld, V_true: np.ndarray, opt_path: list[int],
                args, hp: dict) -> dict:
    signed = np.zeros(args.steps)
    abs_curve = np.zeros(args.steps)
    V_runs, visit_runs = [], []
    enc_for_diag = None
    for r in range(args.runs):
        encoder = build_encoder(kind, env, args.size, args.radius_tiles,
                                args.seed_start + r)
        if enc_for_diag is None:
            enc_for_diag = encoder
        learner, td_curve, visits = train(env, encoder, hp, args.steps, args.epsilon,
                                          args.alpha_decay, args.gamma,
                                          args.seed_start + r)
        signed += td_curve
        abs_curve += np.abs(td_curve)
        V_runs.append(learned_values(learner, encoder, env))
        visit_runs.append(visits)
    signed /= args.runs
    abs_curve /= args.runs
    V_learned = np.mean(V_runs, axis=0)
    visits = np.sum(visit_runs, axis=0)

    # Evaluate value accuracy only on *adequately-visited* states: on the larger 8x8
    # grid several F-tiles sit in hole-walled pockets the eps-soft policy reaches a
    # handful of times, so their value never converges — a coverage limit (documented
    # for the continuous env in the README), not an encoder effect. min_visits floors it.
    visited = np.array([s for s in range(env.n_states)
                        if visits[s] >= args.min_visits and s not in env.terminal])
    abs_err = np.abs(V_learned[visited] - V_true[visited])
    max_err, mean_err = float(abs_err.max()), float(abs_err.mean())
    value_scale = float(np.abs(V_true[visited]).max())
    value_tol = max(0.5, 0.05 * value_scale)

    w = min(2000, args.steps)
    final_residual = float(abs(signed[-w:].mean()))
    final_abs = float(abs_curve[-w:].mean())
    monotonic = is_monotonic([V_learned[s] for s in opt_path[:-1]])

    converged = final_residual < 0.05
    values_correct = max_err < value_tol
    overall = converged and values_correct and monotonic

    diag = overlap_distance_stats(enc_for_diag, env)
    return {
        "encoder": kind,
        "k_active": int(getattr(enc_for_diag, "k", 0)),
        "n_bits": int(enc_for_diag.n_bits),
        "checks": {"converged": converged, "values_correct": values_correct,
                   "gradient_monotonic": monotonic, "overall": overall},
        "metrics": {"final_residual": final_residual, "final_abs_td": final_abs,
                    "max_abs_value_error": max_err, "mean_abs_value_error": mean_err,
                    "value_tol": value_tol, "value_scale": value_scale,
                    "n_visited": int(visited.size)},
        "overlap_distance": diag,
        "V_learned": V_learned.tolist(),
        "path_V_learned": [float(V_learned[s]) for s in opt_path],
        "_signed_curve": signed,    # popped before JSON
        "_abs_curve": abs_curve,
        "_visited": visited,
    }


# -- Main -----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", default="8x8", choices=("4x4", "8x8"))
    p.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=ENCODERS)
    p.add_argument("--size", type=int, default=1024, help="SDR width (bits).")
    p.add_argument("--radius-tiles", type=float, default=2.0,
                   help="Overlap radius (in grid tiles) for the coordinate encoders.")
    p.add_argument("--steps", type=int, default=400_000)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=0.35,
                   help="eps-soft-optimal policy (>0 to cover the larger grid).")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lam", type=float, default=0.6)
    p.add_argument("--alpha-decay", type=float, default=0.0005)
    p.add_argument("--min-visits", type=int, default=200,
                   help="Min visits (summed over runs) for a state to count in the "
                        "value-accuracy check. On 8x8 the top-right F-pocket (s=5,6,7,15) "
                        "is hole-walled and gets <70 visits/2M steps -> coverage-starved; "
                        "the next-covered state has >170, so 200 cleanly excludes it.")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    env = GridWorld(args.map, DEFAULT_REWARDS)
    hp = {"alpha": args.alpha, "gamma": args.gamma, "lam": args.lam}
    V_true = np.array(policy_evaluation(env, args.epsilon, args.gamma))
    opt_path = env.optimal_path(args.gamma)

    print("\n" + "=" * 88)
    print(f"GRID-DISTANCE OVERLAP EXPERIMENT  [{args.map} | {env.n_states} states | "
          f"{args.size}-bit SDRs]")
    print("=" * 88)
    print(f"steps={args.steps} runs={args.runs} eps={args.epsilon} "
          f"alpha={args.alpha} gamma={args.gamma} lambda={args.lam} "
          f"alpha_decay={args.alpha_decay} radius_tiles={args.radius_tiles}")
    print(f"optimal path (len {len(opt_path)}): {opt_path}")
    print("-" * 88)
    hdr = (f"{'encoder':<14}{'k':>5}{'corr(ov,eucl)':>15}{'maxPairOv':>11}"
           f"{'max|V-Vt|':>11}{'mean':>8}{'tol':>7}{'conv':>6}{'val':>5}{'grad':>6}{'OK':>5}")
    print(hdr)

    results = []
    for kind in args.encoders:
        res = run_encoder(kind, env, V_true, opt_path, args, hp)
        results.append(res)
        c = res["checks"]
        m = res["metrics"]
        d = res["overlap_distance"]
        print(f"{kind:<14}{res['k_active']:>5}{d['corr_overlap_euclidean']:>15.3f}"
              f"{d['max_pair_overlap']:>11}{m['max_abs_value_error']:>11.3f}"
              f"{m['mean_abs_value_error']:>8.3f}{m['value_tol']:>7.2f}"
              f"{'Y' if c['converged'] else 'n':>6}{'Y' if c['values_correct'] else 'n':>5}"
              f"{'Y' if c['gradient_monotonic'] else 'n':>6}"
              f"{'PASS' if c['overall'] else 'FAIL':>5}")
    print("=" * 88)

    # Plot before stripping the curves.
    if not args.no_plot:
        plot_path = make_plots(results, V_true, opt_path, env, args)
        print(f"Saved plot    -> {plot_path}")

    # Persist (strip the heavy numpy curves).
    payload = {
        "config": {"map": args.map, "n_states": env.n_states, "size": args.size,
                   "radius_tiles": args.radius_tiles, "steps": args.steps,
                   "runs": args.runs, "epsilon": args.epsilon,
                   "alpha_decay": args.alpha_decay, "min_visits": args.min_visits, **hp,
                   "optimal_path": opt_path},
        "V_true": V_true.tolist(),
        "results": [],
    }
    for res in results:
        clean = {k: v for k, v in res.items() if not k.startswith("_")}
        payload["results"].append(clean)
    out = args.output_dir / f"results_overlap_{args.map}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"Saved metrics -> {out}")


def make_plots(results, V_true, opt_path, env: GridWorld, args) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {"unique": "#9aa0a6", "index-rdse": "#b85c38",
               "coord-scalar": "#0b6e4f", "coord-rdse": "#1f6feb"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    # (1) Overlap vs grid (Chebyshev) distance — the encoder's spatial structure.
    ax = axes[0]
    for res in results:
        by = res["overlap_distance"]["mean_overlap_by_chebyshev"]
        xs = sorted(int(k) for k in by)
        ax.plot(xs, [by[str(x)] if str(x) in by else by[x] for x in xs], "o-",
                color=palette.get(res["encoder"]), lw=2, label=res["encoder"])
    ax.set_title("1. Code overlap vs grid (Chebyshev) distance")
    ax.set_xlabel("Chebyshev distance between tiles")
    ax.set_ylabel("Mean shared active bits")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)

    # (2) Learned vs analytic value (visited states), per encoder.
    ax = axes[1]
    lo = min(V_true[s] for s in range(env.n_states) if s not in env.terminal)
    hi = max(V_true[s] for s in range(env.n_states) if s not in env.terminal)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="y = x (exact)")
    for res in results:
        vis = res["_visited"]
        Vl = np.array(res["V_learned"])
        ax.scatter(V_true[vis], Vl[vis], s=22, alpha=0.7,
                   color=palette.get(res["encoder"]), label=res["encoder"])
    ax.set_title("2. Learned vs analytic V^pi (visited)")
    ax.set_xlabel("Analytic V^pi(s)")
    ax.set_ylabel("Learned V(s)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)

    # (3) Discounted gradient along the optimal path.
    ax = axes[2]
    xs = np.arange(len(opt_path))
    ax.plot(xs, [V_true[s] for s in opt_path], "ks--", lw=1.4, alpha=0.7,
            label="analytic V^pi")
    for res in results:
        ax.plot(xs, res["path_V_learned"], "o-", lw=1.8, alpha=0.85,
                color=palette.get(res["encoder"]), label=res["encoder"])
    ax.set_title("3. Discounted gradient along optimal path (S -> G)")
    ax.set_xlabel("Position on optimal path")
    ax.set_ylabel("V(s)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)

    fig.suptitle(f"Grid-distance-aware overlap on {args.map} FrozenLake "
                 f"({args.size}-bit SDRs) — thesis TD(lambda)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = args.output_dir / f"sparse_repr_overlap_{args.map}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
