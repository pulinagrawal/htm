"""Does high exploration rescue the HTM agent on ContinuousFrozenLake?

Mirrors the finding in scripts/regular_vs_continuous_fzlake/README.md: in the
reset-free continuous lake the goal is a non-absorbing attractor and the start
region is starved of coverage, so a low-exploration agent learns a poor policy
from S; raising exploration (eps 0.1 -> 0.6) rescued tabular SARSA (18% -> 100%).

This agent has no epsilon knob (its exploration is structural in OutputField),
so we inject epsilon-greedy at the EXECUTED action: with prob eps the move is a
uniform-random tile step instead of the brain's chosen action. TD value learning
is state/reward-driven, so this only changes state coverage -- the same lever the
README used.

For each eps we report the training goal/hole trend and a greedy evaluation:
20 rollouts from S (exploration off), max 50 steps, counting goal reaches.

Run:
    .venv/bin/python scripts/fix_b1-6_inconcistencies_test/exploration_sweep.py --steps 4000
"""
import argparse
import contextlib
import io
import logging
import os
import random
import sys

logging.disable(logging.CRITICAL)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import gymnasium as gym

from core.brain import Brain
from core.HTM import ColumnField, InputField, OutputField
from core.sungur import ValueField
from encoder_layer.category import CategoryParametersNew
from gym_adapter import GymBrain
from continuous_frozen_lake import ContinuousFrozenLake, TILES

NS, NA = 16, 4


def build_brain(cpc, ncols, trace_decay=None):
    sp = CategoryParametersNew(size=2048, active_bits_per_category=40, category_list=list(range(NS)))
    state = InputField(encoder_params=sp)
    for s in range(NS):
        state.encoder.encode(s)
    col = ColumnField(input_fields=[state], non_spatial=True, cells_per_column=cpc)
    go = ValueField(input_fields=[col], non_spatial=True, cells_per_column=cpc)
    nogo = ValueField(input_fields=[col], non_spatial=True, cells_per_column=cpc)
    if trace_decay is not None:
        # lambda: per-step trace decay is td_discount * trace_decay; higher => longer
        # credit horizon (less amnesiac), so reward propagates further back from the goal.
        go.trace_decay = trace_decay
        nogo.trace_decay = trace_decay
    col.go_field = go
    col.nogo_field = nogo
    apar = CategoryParametersNew(size=32, active_bits_per_category=5, category_list=list(range(NA)))
    action = OutputField(input_field=col, encoder_params=apar, size=32)
    col.add_input_fields([action])  # ensure action is part of column's input for learning
    for a in range(NA):
        action.encoder.encode(a)
    return Brain({"state": state, "columns": col, "go": go, "nogo": nogo, "action": action})


def greedy_eval(agent, n_rollouts=20, max_steps=50):
    """Roll out from S with exploration off; count goal reaches and avg steps."""
    env = ContinuousFrozenLake(reward_schedule=(1, -1, 0),
                               env=gym.make("FrozenLake-v1", is_slippery=False))
    dn = io.StringIO()
    goals = 0
    steps_to_goal = []
    for _ in range(n_rollouts):
        obs, _ = env.reset()
        last = 0.0
        for t in range(1, max_steps + 1):
            with contextlib.redirect_stdout(dn):
                a = agent.step(obs, reward=last, learn=False)  # no learning, no eps override
            obs, r, term, _ = env.step(a)
            last = float(r)
            if TILES[obs] == "G":
                goals += 1
                steps_to_goal.append(t)
                break
    avg = sum(steps_to_goal) / len(steps_to_goal) if steps_to_goal else float("nan")
    return goals, n_rollouts, avg


def probe_tile_values(brain, go, nogo):
    """Direct per-tile V(s): feed each tile (learn off) and read the weighted
    avg value of the Go and NoGo fields. Prints a 4x4 grid of (Go | NoGo | Net)."""
    dn = io.StringIO()
    print("learned V(s) per tile  (Go | NoGo | Net):", flush=True)
    for row in range(4):
        cells = []
        for coli in range(4):
            s = row * 4 + coli
            with contextlib.redirect_stdout(dn):
                brain.encode_only({"state": float(s)})
                brain.compute_only(learn=False)
            gv, nv = go.avg_value(), nogo.avg_value()
            cells.append(f"{TILES[s]}{s:>2}[{gv:+6.2f}|{nv:+6.2f}|{gv - nv:+6.2f}]")
        print("  " + "  ".join(cells), flush=True)


def run_one(eps, steps, seed, cpc, ncols, window, trace_decay=None):
    random.seed(seed)
    brain = build_brain(cpc, ncols, trace_decay=trace_decay)

    def o2i(o):
        return {"state": float(o)}

    def b2a(b):
        v = b.get("action")
        return int(v) if (v is not None and 0 <= int(v) < NA) else random.randint(0, NA - 1)

    agent = GymBrain(brain, o2i, b2a)
    env = ContinuousFrozenLake(reward_schedule=(1, -1, 0),
                               env=gym.make("FrozenLake-v1", is_slippery=False))
    obs, _ = env.reset()
    last = 0.0
    dn = io.StringIO()
    g = h = 0
    win_g = win_h = 0
    start_visits = 0
    td = brain.fields["go"].td_discount
    lam = brain.fields["go"].trace_decay
    print(f"\n===== epsilon = {eps}  lambda(trace_decay) = {lam}  "
          f"(per-step trace x{td * lam:.3f}) =====", flush=True)
    print("window | goals | holes | goal_share", flush=True)
    for step in range(1, steps + 1):
        with contextlib.redirect_stdout(dn):
            brain_action = agent.step(obs, reward=last, learn=True)
        # epsilon-greedy on the executed action (state-coverage lever)
        action = random.randint(0, NA - 1) if random.random() < eps else brain_action
        obs, r, term, _ = env.step(action)
        last = float(r)
        if obs == 0:
            start_visits += 1
        if term:
            if r > 0:
                g += 1; win_g += 1
            elif r < 0:
                h += 1; win_h += 1
        if step % window == 0:
            tot = win_g + win_h
            share = win_g / tot if tot else 0.0
            print(f"{step:6d} | {win_g:5d} | {win_h:5d} | {share:.0%}", flush=True)
            win_g = win_h = 0
    probe_tile_values(brain, brain.fields["go"], brain.fields["nogo"])
    succ, n, avg = greedy_eval(agent)
    print(f"train totals: goals={g} holes={h}  start(S)-visits={start_visits}", flush=True)
    print(f"GREEDY EVAL: reached goal {succ}/{n} rollouts ({100*succ/n:.0f}%), "
          f"avg steps-to-goal={avg:.1f}", flush=True)
    return eps, succ, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=str, default="0",
                    help="comma-sep seeds; if set, each eps is run over all seeds "
                         "and mean +/- spread of greedy success is reported")
    ap.add_argument("--cpc", type=int, default=2)
    ap.add_argument("--ncols", type=int, default=512)
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--epsilons", type=str, default="0.1,0.6")
    ap.add_argument("--lambdas", type=str, default="0.7",
                    help="comma-sep trace_decay (lambda) values to sweep; default uses "
                         "ValueField's built-in 0.6")
    args = ap.parse_args()

    epsilons = [float(x) for x in args.epsilons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed]
    lambdas = [float(x) for x in args.lambdas.split(",")] if args.lambdas else [None]

    per_key = {}
    for lam in lambdas:
        for eps in epsilons:
            for sd in seeds:
                print(f"\n######## eps={eps}  lambda={lam}  seed={sd} ########", flush=True)
                _, succ, n = run_one(eps, args.steps, sd, args.cpc, args.ncols,
                                     args.window, trace_decay=lam)
                per_key.setdefault((lam, eps), []).append(succ)

    print("\n==== SUMMARY (greedy success from S, n=20 rollouts/seed) ====", flush=True)
    for (lam, eps), succs in per_key.items():
        rates = [100.0 * s / 20 for s in succs]
        mean = sum(rates) / len(rates)
        spread = (max(rates) - min(rates)) if len(rates) > 1 else 0.0
        detail = ", ".join(f"{r:.0f}%" for r in rates)
        lam_str = "default(0.6)" if lam is None else f"{lam}"
        print(f"  lambda={lam_str} eps={eps}: mean {mean:.0f}%  "
              f"(range {min(rates):.0f}-{max(rates):.0f}, spread {spread:.0f}pp)  "
              f"seeds=[{detail}]", flush=True)


if __name__ == "__main__":
    main()
