# B1–B6 reward-circuitry fix — test scripts

Tests for the B1–B6 fixes from `IMPLEMENTATION_INCONSISTENCIES.md`, applied to
`src/core/sungur.py` (`ValueField`) and `src/core/HTM.py` (`ApicalSegment.adapt`).

Run with the project venv (it has `mmh3`, `gymnasium`, etc.):

```bash
.venv/bin/python scripts/fix_b1-6_inconcistencies_test/verify_equations.py
.venv/bin/python scripts/fix_b1-6_inconcistencies_test/frozenlake_learning_test.py --steps 3000
```

## `verify_equations.py`
Deterministic, hand-computed unit checks that each fix matches its thesis
equation. Drives `ValueField` / `ApicalSegment` directly (bypassing
`ColumnField.__init__`) with mock cells/synapses. Exit code 0 iff all pass.

- **B1** state-value weight: predicted active = 10, bursting active = 1 (Eq 5.2)
- **B2** `avg_value` normalized by activation size, not total cells (Eq 5.2)
- **B3** TD error averages `(R + γ·AvgValue − Value_i)` over previous neurons (Eq 5.3)
- **B4** trace order: decay (×γλ) → error → refresh(prev_active→1) → update (Eqs 5.1/5.4/5.5)
- **B5** apical adaptation includes the learning rate α (Eqs 4.2/4.3)
- **B6** opposite-sign TD error weakens prev-active apical synapses (Eq 4.2)

## `frozenlake_learning_test.py`
Builds a small instance of the `agents/frozen_lake.py` brain (same code paths,
smaller so it runs fast), trains on `ContinuousFrozenLake`, and reports:

1. `go`/`nogo` value stats and whether the two fields are identical — a probe for
   **B7** (D1/D2 sharing one error signal). If they are identical, the
   Go−NoGo action signal is zero everywhere and behaviour cannot improve, even
   with B1–B6 correct.
2. Reward / goal / hole trend (first vs last half) — a behaviour-learning signal.
3. A direct per-tile `V(s)` map (Go | NoGo | Net) probed by feeding each tile
   with learning off.

### Findings (as of these fixes)
- `verify_equations.py`: all checks pass — B1–B6 are correct at the equation level.
- `frozenlake_learning_test.py`: the value function learns a sane gradient
  (goal is the least-negative state, holes/distant tiles most negative), **but**
  `go.values == nogo.values` is `True` → Net = 0 everywhere → the agent does not
  yet learn to navigate. The remaining blocker is **B7** (separate D1/D2 error
  selection), not B1–B6.
