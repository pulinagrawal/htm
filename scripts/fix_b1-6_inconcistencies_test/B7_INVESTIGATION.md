# B7 investigation & FrozenLake learning conclusions

This documents the work that started from the B7 item in
`IMPLEMENTATION_INCONSISTENCIES.md` ("D1 and D2 share an identical error
signal") and the chain of experiments it led to while trying to get the
HTM-native agent to actually learn FrozenLake.

**TL;DR**
- **B7 needs no code fix.** The Go/No-Go distinction is already implemented
  correctly in `ColumnField.apical_compute()` via sign-gated apical segments on
  L5. An earlier "B7 still open" claim was a *measurement error* (see below).
- **B1–B6 are correct** (10/10 equation checks) and, with a sparse reward, the
  TD value learning is numerically stable.
- The agent reaches the goal from S in **~54% of greedy rollouts** (5 seeds,
  4000 steps) — real but unreliable, far from solving the task.
- **Neither exploration nor slower trace decay helps.** Both were tested with
  proper multi-seed measurement and produced no behavioural improvement.
- The consistent bottleneck across every experiment is the **value→action
  pathway** (`OutputField` decode + the open Section C voluntary-action
  mechanics), not the reward circuitry.

---

## 1. What B7 actually is, and why it needs no fix

`IMPLEMENTATION_INCONSISTENCIES.md` B7 reads: *"go_field and nogo_field are two
independent ValueFields over the same column field, each computing its own
avg_error from the same reward — so both compute the same scalar."*

Initially this was read as a bug and "confirmed" by checking
`go.values == nogo.values` (they are equal). **That check is wrong.** `go` and
`nogo` are *value estimators* — they are *supposed* to converge to the same
V(s) for a given state (the thesis says D1/D2 share machinery and differ only in
the sign of the error they learn from). Equal value arrays are expected and not
evidence of a bug.

The Go/No-Go behavioural split lives elsewhere — in
`ColumnField.apical_compute()` (`src/core/HTM.py`):

```python
self.select_learning_cells(segments_attr='go_segments',
    segment_factory=lambda cell: ApicalSegment(parent_cell=cell, field=self.go_field,  sign=+1))
self.select_learning_cells(segments_attr='nogo_segments',
    segment_factory=lambda cell: ApicalSegment(parent_cell=cell, field=self.nogo_field, sign=-1))
self.depolarize(segments_attr='apical_segments')   # nets signed scores
```

- `go_segments` (sign +1) sample `go_field`; `nogo_segments` (sign −1) sample
  `nogo_field`.
- `ApicalSegment.adapt` (B5/B6) gates learning by sign: a +1 segment strengthens
  on positive TD error and weakens on negative; a −1 segment does the reverse.
- `Cell.depolarize_apical` nets the signed scores into `go_depolarized` /
  `nogo_depolarized`, which is what the motor layer (`OutputField`) reads.

So the D1/D2 distinction is realized as **sign-gated apical segments on L5**, and
B7 is an *acceptable abstraction*, not a defect. **No code change was made for
B7.**

### Correct B7 probe
The right signal is per-tile **Go vs No-Go depolarization counts** of L5 cells,
not value-array equality. `_probe_depol.py` measures this. In the real-size
brain (2048-cell value fields) it confirms the pathway engages:

```
L5 apical synapses grown:  go = 1599   nogo = 4762
tiles with any go/nogo depolarization: 16/16
```

(In the *shrunken* test harness with ~96-cell value fields, `max_synapses =
int(0.02 × 96) = 1` and `grow()` floors `int((1−0)×0.5) = 0`, so no apical
synapses ever grow and the pathway looks dead. That was a sizing artifact of the
small probe, not a real failure. Use value fields of at least a few hundred
cells.)

---

## 2. The divergence bug found along the way (and its fix)

Running the **real** brain end-to-end revealed a genuine bug: value estimates
grow exponentially and overflow to NaN (~step 1600):

```
step   5:  max|value| 1.3
step 200:  4.7e16
step 800:  2.4e69
step1600:  nan   → OverflowError in calculate_avg_error
```

**Cause:** B1's predicted-weight = 10 combined with B2's normalization. Eq. 5.2
is implemented as `Σ(weightᵢ·valueᵢ) / n` where `n` = count of active cells.
When most active cells are "predicted" (weight 10), `avg_value ≈ 10·mean(value)`
— a ×10 gain. The bootstrap then becomes
`V ← V·(1 + α(γ·10 − 1)) = 5.25·V` per step (α=0.5, γ=0.95) → divergence.
Telltale sign in the logs: `|avg_value|` was consistently ~10× the largest
single value, which a true average can never be.

This matches the literal thesis Eq. 5.2 text, so it is a thesis-equation
instability, not a transcription error. It likely stayed bounded in the thesis
because rewards there were *sparse* (~0 most steps).

**Fix used:** switch the reward schedule from the dense `(10, -10, -1)` to the
sparse `(1, -1, 0)` (goal +1, hole −1, step 0). This removed the per-step driver
of the feedback loop and the values stayed bounded:

```
sparse-reward run:  max|go.value| 3.8 → 10 over 2000 steps (linear creep, no overflow)
```

(A more principled alternative — normalizing by `Σ weights` instead of `n` so
`avg_value ≤ max value` — was identified but not applied, since the reward
change resolved the immediate blocker.)

> Note: setting the Go/No-Go `ValueField`s to `non_spatial=False` *also*
> destabilizes (spatially-pooled value cells stay active across many steps and
> self-amplify through the bootstrap). The stable configuration is
> `non_spatial=True` value fields.

---

## 3. Behavioural experiments (all measured over 5 seeds)

Setup: small instance of the `agents/frozen_lake.py` brain, sparse reward
`(1, -1, 0)`, 4000 training steps, greedy evaluation = 20 rollouts from S with
exploration and learning off, max 50 steps. Metric: % of rollouts reaching the
goal.

### Baseline (ε=0.1, default λ=0.6)
```
seeds: 70%, 55%, 60%, 40%, 45%   →  mean 54%, range 40–70%, spread 30pp
```
Real, learned, goal-directed behaviour (well above the random floor), but
unreliable and high-variance. **A single seed is meaningless here** — the
earlier "70%" was a lucky seed.

### Experiment A — higher exploration (hypothesis: coverage starvation)
Motivated by `scripts/regular_vs_continuous_fzlake/README.md`, where ε 0.1→0.6
took tabular SARSA from 18%→100% on the continuous lake. Injected ε-greedy on
the *executed* action (state-coverage only; TD learning untouched).

```
ε=0.1: mean 54%        ε=0.6: 40% (single seed, within the 30pp noise band)
start(S)-visits barely moved (264 vs 266)
```
**No benefit.** The README's mechanism (goal-camping → start starvation) does
**not** apply: this agent *under*-exploits (272 goals vs 989 holes — it wanders
and dies), so adding randomness to an already-diffuse agent changes nothing.

### Experiment B — slower trace decay (hypothesis: "too amnesiac")
λ (trace_decay) swept over {0.6, 0.9, 0.99} × 5 seeds at ε=0.1. Per-step trace
factor = `td_discount × λ` = 0.57 / 0.855 / 0.941.

```
λ=0.6 : 70,55,60,40,45  → mean 54%
λ=0.9 : 70,55,60,40,45  → mean 54%
λ=0.99: 70,55,60,40,45  → mean 54%   (identical greedy results)
```
The value *maps* do differ across λ (e.g. F14: −1.96 / −2.67 / −3.31), so λ is
genuinely changing the values — but the **behaviour is identical**. Slower decay
does not help.

### A worrying observation about the value function
At 4000 steps the value gradient is **wrong and degrades with training**: the
goal G15 ends up among the *most negative* tiles (−3.88 to −4.54), and higher λ
makes it *more* negative. Yet at 1500 steps an earlier run had G15 **positive
(+1.95)** with a correct gradient (bottom row climbing toward the goal). So V(s)
is **not converging — it drifts negative/inverts** over training. The behaviour
score (~54%) is decoupled from this: it stays flat whether the values look right
or wrong.

---

## 4. Conclusions

1. **B7 is correctly implemented already** via sign-gated apical segments in
   `apical_compute()`. No fix required. The "go==nogo values" check was the
   wrong probe; use Go/No-Go depolarization counts (`_probe_depol.py`).
2. **B1–B6 are correct** and, with a sparse reward, numerically stable.
3. **The reward circuitry is not the bottleneck.** Three independent levers
   (exploration, trace decay, and the value maps themselves) all point the same
   way: changing the values does not change the behaviour.
4. **The bottleneck is the value→action pathway** — `OutputField`'s
   decode-from-probabilities action selection plus the still-open **Section C**
   voluntary-action mechanics (C1 distal∧apical coincidence, C2 motor top-3
   WTA). Correct or incorrect values are not being translated into goal-directed
   action.
5. **The value function does not converge** at this horizon (drifts/inverts) —
   worth investigating alongside Section C, possibly related to the weight-10
   bootstrap gain even under the sparse reward.

### Recommended next step
Instrument a single greedy rollout and log, per step: Go/No-Go depolarization
counts, the `OutputField` action probabilities, the action chosen, and the
action the value map would imply — to pinpoint *where* the values stop steering
the action.

---

## 5. Scripts in this folder

| file | purpose |
|---|---|
| `verify_equations.py` | Deterministic B1–B6 equation checks (10/10 pass). |
| `frozenlake_learning_test.py` | Small-brain training + per-tile V(s) probe (uses sparse reward). |
| `real_brain_learning_curve.py` | Behavioural curve for the real `build_agent()` brain. |
| `exploration_sweep.py` | ε and λ sweeps, multi-seed (`--seeds`, `--epsilons`, `--lambdas`), greedy eval + value map. |
| `_probe_depol.py` | Correct B7 probe: per-tile Go/No-Go depolarization counts + action-prob spread. |
| `README.md` | Overview of the B1–B6 test scripts. |

Run anything with the project venv (has `mmh3`, `gymnasium`):
```bash
.venv/bin/python scripts/fix_b1-6_inconcistencies_test/exploration_sweep.py \
    --steps 4000 --epsilons 0.1 --lambdas 0.6,0.9,0.99 --seeds 0,1,2,3,4
```
