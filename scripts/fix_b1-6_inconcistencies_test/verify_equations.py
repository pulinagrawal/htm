"""Deterministic equation-level checks for the B1-B6 reward-circuitry fixes.

Drives `ValueField` (sungur.py) and `ApicalSegment` (HTM.py) with hand-computed
inputs and asserts the outputs match the thesis equations referenced in
IMPLEMENTATION_INCONSISTENCIES.md.

Run:
    .venv/bin/python scripts/fix_b1-6_inconcistencies_test/verify_equations.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from core.sungur import ValueField
from core.HTM import ApicalSegment, PERMANENCE_INC

PASS = []
def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


# --- minimal stand-ins for the cell state ValueField reads -------------------
class Cell:
    def __init__(self):
        self.active = False
        self.prev_active = False
        self.prev_predictive = False

def make_vf(n):
    """Build a ValueField without running ColumnField.__init__."""
    vf = object.__new__(ValueField)
    vf.cells = [Cell() for _ in range(n)]
    vf._field = vf
    vf.values = [0.0] * n
    vf.traces = [0.0] * n
    vf.td_learning_rate = 0.5
    vf.td_discount = 0.95
    vf.trace_decay = 0.6
    vf.avg_error = 0.0
    vf._weight_fn = ValueField._default_weight
    return vf, vf.cells


# ===== B1 + B2: predicted=10 / burst=1 weighting, normalized by activation ====
vf, c = make_vf(4)
c[0].active = True; c[0].prev_predictive = True   # correctly predicted -> w=10
c[1].active = True; c[1].prev_predictive = False  # bursting             -> w=1
vf.values = [2.0, 1.0, 5.0, 9.0]                  # cells 2,3 inactive -> ignored
# Eq 5.2: (10*2 + 1*1) / n_active(=2) = 21/2 = 10.5
av = vf.avg_value()
check("B1 predicted=10/burst=1  +  B2 normalize by activation size",
      abs(av - 10.5) < 1e-9, f"avg_value={av} (expect 10.5)")

# ===== B3: TD error uses per-neuron previous values =====
vf, c = make_vf(3)
c[2].active = True; c[2].prev_predictive = True   # current state -> avg_value=10*10=100
c[0].prev_active = True; c[1].prev_active = True   # previous-state neurons
vf.values = [2.0, 4.0, 10.0]
vf.calculate_avg_error(reward=1.0)
expect = ((1 + 0.95*100 - 2.0) + (1 + 0.95*100 - 4.0)) / 2          # = 93.0
check("B3 error averages (R + g*AvgValue - Value_i) over prev neurons",
      abs(vf.avg_error - expect) < 1e-9, f"err={vf.avg_error} (expect {expect})")

vf2, c2 = make_vf(3)
c2[2].active = True; c2[2].prev_predictive = True
c2[0].prev_active = True; c2[1].prev_active = True
vf2.values = [0.0, 0.0, 10.0]                                       # change prev values only
vf2.calculate_avg_error(reward=1.0)
check("B3 error responds to per-neuron prev value (not a collapsed constant)",
      abs(vf2.avg_error - (1 + 0.95*100 - 0.0)) < 1e-9, f"err={vf2.avg_error}")

# ===== B4: decay (xGL) -> error -> refresh(prev_active->1) -> update =====
vf, c = make_vf(3)
c[0].prev_active = True                            # state we are leaving
c[1].active = True                                 # current state (burst, w=1)
vf.values = [1.0, 2.0, 0.0]
vf.traces = [0.5, 0.5, 0.5]
gl = vf.td_discount * vf.trace_decay               # 0.95*0.6 = 0.57
vf.update_values(reward=1.0)
expect_traces = [1.0, 0.5*gl, 0.5*gl]              # cell0 refreshed to 1, others decayed
check("B4 traces: decay x(g*l) then refresh prev_active->1",
      all(abs(a-b) < 1e-9 for a, b in zip(vf.traces, expect_traces)),
      f"traces={vf.traces} expect {expect_traces}")
check("B4 value update uses refreshed trace (cell0 gets full alpha*err update)",
      abs((vf.values[0] - 1.0) - 0.5*vf.avg_error*1.0) < 1e-9,
      f"value0={vf.values[0]} err={vf.avg_error}")


# ===== B5 + B6: ApicalSegment.adapt (alpha factor + opposite-sign weakening) ==
class FieldStub:
    def __init__(self, err): self.avg_error = err; self.td_learning_rate = 0.5
class SCell:
    def __init__(self, pa): self.prev_active = pa
class Syn:
    def __init__(self, pa, perm=0.5): self.source_cell = SCell(pa); self.permanence = perm
    def _adjust_permanence(self, increase, strength=1.0):
        if increase: self.permanence = min(1.0, self.permanence + PERMANENCE_INC*strength)
        else:        self.permanence = max(0.0, self.permanence - PERMANENCE_INC*strength)

def make_seg(sign, err):
    seg = object.__new__(ApicalSegment)
    seg.sign = sign; seg.field = FieldStub(err)
    seg.synapses = [Syn(pa=True), Syn(pa=False)]
    return seg

# B5: D1, +error -> strengthen prev-active by inc = sign*err*alpha = 0.2 -> +0.20*0.2 = 0.04
seg = make_seg(+1, +0.4); seg.adapt()
check("B5 alpha factor in apical increment",
      abs(seg.synapses[0].permanence - 0.54) < 1e-9,
      f"perm={seg.synapses[0].permanence} (expect 0.54)")
check("B5 non-prev-active synapse untouched",
      abs(seg.synapses[1].permanence - 0.5) < 1e-9,
      f"perm={seg.synapses[1].permanence} (expect 0.50)")

# B6: D1, -error (opposite sign) -> weaken prev-active by dec = |err|*alpha*2 = 0.4 -> -0.08
seg = make_seg(+1, -0.4); seg.adapt()
check("B6 opposite-sign error weakens prev-active synapse (was a no-op before)",
      abs(seg.synapses[0].permanence - 0.42) < 1e-9,
      f"perm={seg.synapses[0].permanence} (expect 0.42)")

# B6: D2, -error -> strengthen (D2 learns on negative error)
seg = make_seg(-1, -0.4); seg.adapt()
check("B6 D2 strengthens on negative error",
      abs(seg.synapses[0].permanence - 0.54) < 1e-9,
      f"perm={seg.synapses[0].permanence} (expect 0.54)")

print(f"\nRESULT: {sum(PASS)}/{len(PASS)} checks passed")
sys.exit(0 if all(PASS) else 1)
