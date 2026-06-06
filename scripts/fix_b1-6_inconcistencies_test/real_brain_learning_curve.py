"""Behavioural learning curve for the REAL agents/frozen_lake.py brain.

Uses build_agent() unchanged (2048-cell value fields, the real OutputField
action selection) so the result reflects the actual agent, not a shrunken probe.
Reports reward-per-step and goal/hole counts in windows over training.

Run:
    .venv/bin/python scripts/fix_b1-6_inconcistencies_test/real_brain_learning_curve.py --steps 6000
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

from agents.frozen_lake import build_agent
from continuous_frozen_lake import ContinuousFrozenLake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=1000)
    args = ap.parse_args()
    random.seed(args.seed)

    agent = build_agent()
    env = ContinuousFrozenLake(reward_schedule=(10, -10, -1),
                               env=gym.make("FrozenLake-v1", is_slippery=False))

    obs, _ = env.reset()
    last = 0.0
    dn = io.StringIO()

    win = args.window
    win_reward = 0.0
    win_goals = 0
    win_holes = 0
    rewards = []
    nonrandom_actions = 0

    print(f"window={win} steps | reward/step | goals | holes | (cumulative)")
    cum_g = cum_h = 0
    for step in range(1, args.steps + 1):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            a = agent.step(obs, reward=last, learn=True)
        if "NON-RANDOM" in buf.getvalue():
            nonrandom_actions += 1
        obs, r, term, _ = env.step(a)
        last = float(r)
        rewards.append(last)
        win_reward += last
        if term:
            if r > 0:
                win_goals += 1; cum_g += 1
            elif r < -1:
                win_holes += 1; cum_h += 1
        if step % win == 0:
            print(f"  step {step:6d}      | {win_reward/win:+8.3f}  | {win_goals:5d} | {win_holes:5d} | "
                  f"(g={cum_g} h={cum_h})")
            win_reward = 0.0; win_goals = 0; win_holes = 0

    n = len(rewards)
    q = max(1, n // 4)
    print("\n==== SUMMARY ====")
    print(f"steps={n}")
    print(f"reward/step  first quarter {sum(rewards[:q])/q:+.3f}  |  last quarter {sum(rewards[-q:])/q:+.3f}")
    print(f"non-random (value-driven) actions: {nonrandom_actions}/{n} ({100*nonrandom_actions/n:.1f}%)")
    print(f"total goals={cum_g}  total holes={cum_h}")


if __name__ == "__main__":
    main()
