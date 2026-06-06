"""Measure the signal that actually drives behaviour: per-tile Go vs NoGo
depolarization of L5 cells, and the action-probability spread.

This is the correct B7 probe: the Go/NoGo distinction lives in the sign-gated
apical segments on L5 (go_segments sign +1 read go_field, nogo_segments sign -1
read nogo_field), not in the value fields being unequal.
"""
import argparse, contextlib, io, logging, os, random, sys
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
ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--cpc", type=int, default=2)
ap.add_argument("--ncols", type=int, default=0, help="0 => non_spatial value fields")
args = ap.parse_args(); random.seed(0)

sp = CategoryParametersNew(size=48, active_bits_per_category=6, category_list=list(range(NS)))
state = InputField(encoder_params=sp)
for s in range(NS): state.encoder.encode(s)
col = ColumnField(input_fields=[state], non_spatial=True, cells_per_column=args.cpc)
if args.ncols > 0:
    go = ValueField(input_fields=[col], non_spatial=False, num_columns=args.ncols, cells_per_column=args.cpc)
    nogo = ValueField(input_fields=[col], non_spatial=False, num_columns=args.ncols, cells_per_column=args.cpc)
else:
    go = ValueField(input_fields=[col], non_spatial=True, cells_per_column=args.cpc)
    nogo = ValueField(input_fields=[col], non_spatial=True, cells_per_column=args.cpc)
col.go_field = go; col.nogo_field = nogo
apar = CategoryParametersNew(size=32, active_bits_per_category=5, category_list=list(range(NA)))
action = OutputField(input_field=col, encoder_params=apar, size=32)
for a in range(NA): action.encoder.encode(a)
brain = Brain({"state": state, "columns": col, "go": go, "nogo": nogo, "action": action})

def o2i(o): return {"state": float(o)}
def b2a(b):
    v = b.get("action")
    return int(v) if (v is not None and 0 <= int(v) < NA) else random.randint(0, NA - 1)
agent = GymBrain(brain, o2i, b2a)
env = ContinuousFrozenLake(reward_schedule=(10, -10, -1), env=gym.make("FrozenLake-v1", is_slippery=False))

obs, _ = env.reset(); last = 0.0; dn = io.StringIO()
for _ in range(args.steps):
    with contextlib.redirect_stdout(dn):
        a = agent.step(obs, reward=last, learn=True)
    obs, r, term, _ = env.step(a); last = float(r)

# total apical synapses grown on L5, per polarity
go_syn = sum(len(seg.synapses) for c in col.cells for seg in c.go_segments)
nogo_syn = sum(len(seg.synapses) for c in col.cells for seg in c.nogo_segments)
print(f"L5 apical synapses grown:  go_segments={go_syn}  nogo_segments={nogo_syn}")

print("\nPer-tile L5 depolarization + action-prob spread (learn off):")
print("tile | #go_depol | #nogo_depol | action probs [L,D,R,U] | argmax")
any_bias = False
for s in range(NS):
    with contextlib.redirect_stdout(dn):
        brain.encode_only({"state": float(s)})
        brain.compute_only(learn=False)
        col.apical_compute(learn=False)        # set go/nogo depolarization from learned segments
        probs = action.activation_probabilities()
    ng = sum(1 for c in col.cells if c.go_depolarized)
    nn = sum(1 for c in col.cells if c.nogo_depolarized)
    # collapse 32 output-cell probs to 4 action scores via decode candidates
    scores = []
    for av in range(NA):
        enc = action.encoder._encoding_cache.get(av) or action.encoder.register_encoding(av)
        scores.append(sum(p for p, b in zip(probs, enc) if b))
    spread = max(scores) - min(scores)
    if spread > 1e-9 or ng or nn:
        any_bias = True
    print(f"  {TILES[s]}{s:>2} |   {ng:>4}   |    {nn:>4}    | "
          f"[{scores[0]:.2f},{scores[1]:.2f},{scores[2]:.2f},{scores[3]:.2f}] | {max(range(NA), key=lambda i: scores[i])}")
print(f"\nany Go/NoGo depolarization or action bias seen? {any_bias}")
