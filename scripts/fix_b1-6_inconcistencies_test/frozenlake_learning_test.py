"""End-to-end FrozenLake learning probe for the HTM-native reward circuitry.

Builds a small instance of the same brain as agents/frozen_lake.py (ColumnField
L5 + Go/NoGo ValueField TD circuitry + OutputField voluntary action), trains it
on ContinuousFrozenLake, then reports:

  * raw go/nogo value stats and whether the two fields are identical (B7 probe),
  * the reward/goal/hole trend (early vs late) as a behaviour-learning signal,
  * a direct per-tile V(s) value map (Go | NoGo | Net).

Small enough to run a few thousand steps in seconds.

Run:
    .venv/bin/python scripts/fix_b1-6_inconcistencies_test/frozenlake_learning_test.py --steps 3000
"""
import argparse
import contextlib
import io
import logging
import os
import random
import sys

logging.disable(logging.CRITICAL)  # silence the encoder's per-step INFO logs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import gymnasium as gym

from core.brain import Brain
from core.HTM import ColumnField, InputField, OutputField
from core.sungur import ValueField
from encoder_layer.category import CategoryParametersNew
from gym_adapter import GymBrain
from continuous_frozen_lake import ContinuousFrozenLake, TILES

NUM_STATES, NUM_ACTIONS = 16, 4


def build_small_brain(state_size, state_bits, cpc):
    sp = CategoryParametersNew(size=state_size, active_bits_per_category=state_bits,
                               category_list=list(range(NUM_STATES)))
    state = InputField(encoder_params=sp)
    for s in range(NUM_STATES):
        state.encoder.encode(s)
    col = ColumnField(input_fields=[state], non_spatial=True, cells_per_column=cpc)
    go = ValueField(input_fields=[col], non_spatial=False, num_columns=2048, cells_per_column=cpc)
    nogo = ValueField(input_fields=[col], non_spatial=False, num_columns=2048, cells_per_column=cpc)
    col.go_field = go
    col.nogo_field = nogo
    apar = CategoryParametersNew(size=32, active_bits_per_category=5,
                                 category_list=list(range(NUM_ACTIONS)))
    action = OutputField(input_field=col, encoder_params=apar, size=32)
    for a in range(NUM_ACTIONS):
        action.encoder.encode(a)
    return Brain({"state": state, "columns": col, "go": go, "nogo": nogo, "action": action})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpc", type=int, default=2)
    ap.add_argument("--state-size", type=int, default=48)
    ap.add_argument("--state-bits", type=int, default=6)
    args = ap.parse_args()
    random.seed(args.seed)

    brain = build_small_brain(args.state_size, args.state_bits, args.cpc)
    go, nogo = brain.fields["go"], brain.fields["nogo"]

    def o2i(o):
        return {"state": float(o)}

    def b2a(b):
        v = b.get("action")
        return int(v) if (v is not None and 0 <= int(v) < NUM_ACTIONS) else random.randint(0, NUM_ACTIONS - 1)

    agent = GymBrain(brain, o2i, b2a)
    env = ContinuousFrozenLake(reward_schedule=(1, -1, 0),
                               env=gym.make("FrozenLake-v1", is_slippery=False))

    obs, _ = env.reset()
    last = 0.0
    rewards = []
    half = args.steps // 2
    ge = he = gl = hl = 0
    dn = io.StringIO()

    for step in range(args.steps):
        with contextlib.redirect_stdout(dn):
            a = agent.step(obs, reward=last, learn=True)
        obs, r, term, _ = env.step(a)
        last = float(r)
        rewards.append(last)
        if term:
            first = step < half
            if r > 0:
                ge += first; gl += (not first)
            elif r < -1:
                he += first; hl += (not first)

    def stats(vals):
        nz = sum(1 for v in vals if abs(v) > 1e-9)
        return f"n={len(vals)} nonzero={nz} min={min(vals):+.3f} max={max(vals):+.3f} mean={sum(vals)/len(vals):+.3f}"

    third = max(1, len(rewards) // 3)
    print("=== value fields ===")
    print("go.values   ", stats(go.values))
    print("nogo.values ", stats(nogo.values))
    print("go.values == nogo.values ?",
          all(abs(g - n) < 1e-9 for g, n in zip(go.values, nogo.values)),
          "  (True => B7 still open: D1/D2 share one error signal)")

    print("\n=== behaviour trend ===")
    print(f"reward first third {sum(rewards[:third])/third:+.3f} | last third {sum(rewards[-third:])/third:+.3f}")
    print(f"goals first/last half {ge}/{gl}   holes first/last half {he}/{hl}")

    print("\n=== direct V(s) probe per tile  (Go | NoGo | Net) ===")
    for row in range(4):
        cells = []
        for coli in range(4):
            s = row * 4 + coli
            with contextlib.redirect_stdout(dn):
                brain.encode_only({"state": float(s)})
                brain.compute_only(learn=False)
            gv, nv = go.avg_value(), nogo.avg_value()
            cells.append(f"{TILES[s]}[{gv:+5.1f}|{nv:+5.1f}|{gv-nv:+5.1f}]")
        print("  " + " ".join(cells))


if __name__ == "__main__":
    main()
