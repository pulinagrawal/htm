# Sparse-representation TD-learning sanity test

**Question.** Does the basic temporal-difference learning principle still hold when
(a) a state is a **sparse representation** (an SDR — a fixed set of active "neurons")
instead of a single tabular index, and (b) value is learned with the **thesis
reward-circuitry formulation** (Eqs 5.1–5.5 of
[`HTM_TD_Agent_Implementation_Reference.md`](../../HTM_TD_Agent_Implementation_Reference.md))?

**Answer: yes.** With all HTM machinery removed, the thesis TD(λ) value update over a
sparse code converges to the *exact* analytic Bellman values, drives the TD error to
zero in expectation, and builds the correct discounted value gradient toward the goal.

This isolates the value math (the B1–B6 fixes in `src/core/sungur.py::ValueField`)
from the HTM stack: a passing run proves the TD formulation itself is sound,
independent of Spatial Pooler / Temporal Memory / D1–D2 wiring / voluntary action.

> **Now also: control.** Building the thesis **D1/D2 Go/No-Go reward circuitry +
> voluntary action** on top of this value substrate (`sparse_control.py`,
> `run_control.py`) **solves episodic FrozenLake at 100% greedy success / 6 steps
> (optimal)** — for all three encoders. See [§ Control](#control-does-the-thesis-action-selection-solve-the-task) below.

## What is and isn't included

This is the **value-only, fixed-policy** slice (the thesis D1/D2 layers learn a
*state value* V(s), not action values — so there is no control/policy-learning here).

| Kept (identical to the thesis / `ValueField`) | Removed (the HTM complexity under test elsewhere) |
|---|---|
| Per-neuron stored `value` + eligibility `trace` | Spatial Pooler / Temporal Memory |
| Eq 5.1 trace decay ×γλ | Distal/apical prediction (depolarization) |
| Eq 5.2 weighted-mean state value | D1/D2 striatal layers + their wiring |
| Eq 5.3 TD error over previous-active neurons | Voluntary action / motor output |
| Eq 5.4 replacing traces (prev-active → 1) | Representation *learning* (SDRs are fixed) |
| Eq 5.5 neuron-value update α·err·trace | The predicted-vs-burst weight (10 vs 1)¹ |
| Per-cycle order (Alg 9 ll.15–19) | |

¹ With no Temporal Memory there is no prediction signal, so every active neuron is
"bursting" → weight 1 and V(s) is the plain mean of active-neuron values. Reintroduce
`burst_weight`/a predicted weight only if you add a prediction signal back.

## Files

- `sparse_encoder.py` — the state→SDR codes, selectable via `--encoder`:
  - `UniqueSDREncoder` (`unique`, default): distinct **non-overlapping** k-of-M SDRs (no
    shared bits) — zero generalization, the cleanest decoupling; one-hot is the k=1 case.
  - `CategoryEncoderAdapter` (`category`): wraps the project's real
    `src/encoder_layer/category.py::CategoryEncoderNew`, which also allocates disjoint
    bit blocks per category → non-overlapping, so it should match `unique`/tabular.
  - `RDSEEncoderAdapter` (`rdse`): wraps the real
    `src/encoder_layer/rdse.py::RandomDistributedScalarEncoder`, encoding the state
    *index as a scalar* → numerically-adjacent states **share bits** (overlap knob
    `--rdse-resolution`). The interference stress-test.
- `sparse_td.py` — `ThesisTDValue`: a line-for-line port of `sungur.ValueField` reduced
  to Eqs 5.1–5.5 over a flat array of neuron values + traces. Never sees the
  environment — only sparse codes.
- `environment.py` — the 4×4 FrozenLake in **two** forms, same map/dynamics:
  - **episodic** — a self-contained tabular model (G/H terminal, reset to S);
  - **continuous** — the *real* `src/continuous_frozen_lake.py::ContinuousFrozenLake`
    (no terminals, no resets), the exact env used by `regular_vs_continuous_fzlake/`.

  Plus a fixed ε-soft-optimal policy and exact `policy_evaluation` (episodic *and*
  continuous variants) for the **ground-truth** V^π the learner must match.
- `run_sanity.py` — runs the fixed policy, checks the three TD principles against the
  analytic ground truth, plots, writes `results*.json`.
- `sparse_control.py` — `GoNoGoActor`: the thesis **control** logic (D1/D2 Go/No-Go
  reward circuitry + learned one-step model + voluntary action + motor WTA) on the sparse
  substrate. See [§ Control](#control-does-the-thesis-action-selection-solve-the-task).
- `run_control.py` — trains `GoNoGoActor` on episodic FrozenLake, reports greedy success,
  learning curve, value gradient; writes `results_control*.json` / `sparse_control*.png`.
- `grid_env.py` — generic NxN FrozenLake (loads the standard gym **8×8** map, 64 states)
  with value-iteration optimal policy/path + analytic `policy_evaluation`. The *larger*
  env for the grid-distance-overlap experiment (see the **Grid-distance-aware overlap**
  section below).
- `coordinate_encoder.py` — `CoordinateEncoder`: encodes a tile's **(row, col)** with a
  real project encoder per axis (`scalar` or `rdse`) and concatenates → 1024-bit SDRs whose
  overlap falls off with grid (diagonal) distance.
- `run_overlap_experiment.py` — runs the grid-distance-overlap comparison
  (`unique` / `index-rdse` / `coord-scalar` / `coord-rdse`) on the 8×8 env at 1024 bits,
  with an overlap-vs-distance diagnostic; writes `results_overlap_8x8.json` /
  `sparse_repr_overlap_8x8.png`.

> **Stacking the HTM layers back on.** Two follow-on experiments add the next HTM layers
> on top of this validated encoder→value substrate, each in its own self-contained folder:
> [`../spatialpooler_td-learning/`](../spatialpooler_td-learning/) (encoder → **Spatial
> Pooler** → value) and [`../tm_td-learning/`](../tm_td-learning/) (encoder → SP →
> **Temporal Memory** → value). Both confirm the thesis TD(λ) update still recovers V^π on
> the deeper representation.

## Run

```bash
# Episodic, default: deterministic optimal-path policy (eps=0) — crisp demonstration.
uv run python scripts/sparse_repr_td-learning/run_sanity.py

# Episodic, stochastic coverage (whole grid; needs step-size decay to converge):
uv run python scripts/sparse_repr_td-learning/run_sanity.py \
    --steps 120000 --epsilon 0.2 --alpha-decay 0.002

# Continuous (reset-free ContinuousFrozenLake; goal is a non-absorbing attractor).
# eps/alpha-decay default to coverage-friendly values automatically here.
uv run python scripts/sparse_repr_td-learning/run_sanity.py --env continuous --steps 200000

# Swap the encoder for the project's real ones (--encoder unique|category|rdse):
uv run python scripts/sparse_repr_td-learning/run_sanity.py --encoder category \
    --epsilon 0.2 --alpha-decay 0.002 --steps 120000
uv run python scripts/sparse_repr_td-learning/run_sanity.py --encoder rdse \
    --rdse-size 256 --rdse-resolution 1.0 --epsilon 0.2 --alpha-decay 0.002 --steps 120000

# CONTROL: thesis D1/D2 Go/No-Go + voluntary action — does it *solve* FrozenLake?
uv run python scripts/sparse_repr_td-learning/run_control.py
uv run python scripts/sparse_repr_td-learning/run_control.py --encoder rdse --episodes 3000
```

`unique`+episodic outputs `sparse_repr_td.png` / `results.json`; other configs are
suffixed (`_continuous`, `_category`, `_rdse`).

## The three checks ("do TD principles hold?")

1. **Convergence** — the **expected** TD error (signed-mean = Bellman residual) → 0.
   We track this rather than `mean |TD error|` because, under a *stochastic* policy,
   the instantaneous `|TD error|` only has zero *expectation* — it floors out at the
   target's standard deviation. The signed mean → 0 in **both** the deterministic and
   stochastic regimes; that is the universal convergence statement.
2. **Correct values** — learned V(s) converges to the exact analytic V^π(s) from policy
   evaluation on the same model (`max |V_learned − V^π| < 0.5`). Confirms the update is
   *unbiased*, not just convergent to some arbitrary fixed point.
3. **Discounted gradient** — V rises monotonically along the optimal path `0→4→8→9→13→14→15`
   and the step-to-step structure follows γ-discounting toward the goal reward.

## Findings

**Deterministic policy (`eps=0`, default; 40k steps × 5 seeds, α=0.5, γ=0.95, λ=0.6):**
every check passes and the learned values match the closed-form discounted returns to
four decimals — i.e., the sparse code recovers tabular TD behaviour exactly.

| state | tile | V_learned | V_true (analytic) |
|--:|:--:|--:|--:|
| 0 | S | 3.213 | 3.213 |
| 4 | F | 4.435 | 4.435 |
| 8 | F | 5.721 | 5.721 |
| 9 | F | 7.075 | 7.075 |
| 13 | F | 8.500 | 8.500 |
| 14 | F | 10.000 | 10.000 |
| 15 | G | 0.000 (terminal) | 0.000 |

`|E[TD error]|: 0.019 → 0.000`,  `max |V_learned − V^π| = 0.000`,  gradient monotonic.

**Stochastic policy (`eps=0.2 + --alpha-decay 0.002`; 120k steps × 5 seeds):** with the
whole grid now visited, the learned values still converge to V^π
(`max |V_learned − V^π| ≈ 0.29`, mean ≈ 0.11) and the signed TD residual → ≈0, while
`mean |TD error|` correctly plateaus at its irreducible noise floor (~1.7). A constant
step size leaves residual variance, so Robbins–Monro `--alpha-decay` is needed here —
this is standard TD-with-function-approximation behaviour, not a defect.

**Continuous env (`--env continuous`, the reset-free `ContinuousFrozenLake`; ε=0.4 +
α-decay, 200k × 5):** all three checks pass here too. The thesis update — now with **no
terminal bootstrap cut and no resets**, traces persisting across the whole stream —
converges to the exact *continuous* V^π on every (now well-covered) state
(`max |V_learned − V^π| ≈ 2.3`, mean ≈ 1.2, well within the 5%-of-scale tolerance), with
the signed residual → ≈0. As expected for the non-absorbing goal, values are larger
(O(60) here vs O(10) episodic) and `mean |TD error|` floors at ~11 — the goal-as-attractor
and bigger value scale documented in `regular_vs_continuous_fzlake/`. The monotonic
discounted gradient toward the goal is preserved.

| | episodic (ε=0) | episodic (ε=0.2) | continuous (ε=0.4) |
|---|--:|--:|--:|
| signed residual → | 0.000 | ≈0 | 0.013 |
| `mean\|TD err\|` floor | 0.000 | ~1.7 | ~11 |
| `max\|V−V^π\|` (visited) | 0.000 | 0.29 | 2.28 |
| value scale | 10 | 10 | 63 |
| all 3 checks | PASS | PASS | PASS |

Takeaway: removing episode boundaries changes the *task* (goal becomes an attractor,
values inflate, coverage needs more exploration) but **not** the soundness of the thesis
TD update — it still converges to the correct Bellman values wherever the agent visits.

### Swapping in the project's real encoders (`--encoder category|rdse`)

The thesis value update is **linear in the SDR features** (V(s) = mean of the active
neurons' values), so its accuracy is governed by one thing: *does the encoder keep
value-distinct states linearly separable?*

| encoder (episodic, ε=0.2 + α-decay, 120k×5) | code overlap | `max\|V−V^π\|` | mean | verdict |
|---|---|--:|--:|---|
| `unique` (baseline) | none | 0.29 | 0.11 | PASS |
| `category` (real `CategoryEncoderNew`) | none (disjoint blocks) | 0.29 | 0.11 | PASS |
| `rdse` res=0.125 (≈1/k, near-disjoint) | max pair overlap 1 | 0.29 | 0.10 | PASS |
| `rdse` res=1.0 (default) | max pair overlap 7; 84/120 pairs share bits | **2.21** | 0.50 | **FAIL** |
| `rdse` size=24 (forced collisions) | severe aliasing | **5.45** | 1.51 | **FAIL** |

Reading this:

- **CategoryEncoder ≡ tabular.** It allocates each state a disjoint bit block, so it
  reproduces the `unique`/tabular result exactly. The thesis TD update drives the real
  project encoder correctly. ✓
- **RDSE holds when overlap is small, degrades when it isn't.** RDSE encodes the *state
  index as a scalar*, so it makes numerically-adjacent FrozenLake states share bits — but
  index-adjacency is **not** value-adjacency here (state 4 is on the optimal path, state 5
  is a hole). At coarse resolution (res=1.0) that shared structure couples value-distinct
  states through their common neurons, biasing the learned values (max err 2.2). Shrinking
  the code (size=24) forces more aliasing and makes it worse (5.5).
- **It's the overlap, not RDSE.** Drop RDSE's resolution to ≈1/k (near-disjoint codes) and
  it recovers the exact tabular result (0.29). So the failure is a **representation/task
  mismatch** — applying a *metric* scalar encoder to a state whose index carries no metric
  meaning — not a failure of the TD math.
- **The TD *principle* still holds even when RDSE fails the accuracy bar.** In every RDSE
  case the signed TD residual → 0 (it converges) and the value gradient along the optimal
  path stays monotonic toward the goal — the policy-relevant ordering survives; only the
  absolute values are biased by feature overlap. And under the deterministic ε=0 policy
  (only the 7 path states visited, whose codes are linearly independent) *all* encoders —
  RDSE included — recover the values exactly.

**Takeaway:** basic TD-learning holds over a sparse code **iff the code separates states
that have different values.** Non-overlapping encoders (`unique`, `category`) always do;
overlapping ones (`rdse`) do only when their similarity structure is aligned with — or at
least not actively misaligned against — the value function.

### Why continuous needs exploration (consistency with regular_vs_continuous)

`regular_vs_continuous_fzlake/` found that the continuous setting needs exploration for
the agent to *work*; the same coverage starvation appears here, just in value-evaluation
form. Run it deterministically to see it:

```bash
uv run python scripts/sparse_repr_td-learning/run_sanity.py \
    --env continuous --epsilon 0 --alpha-decay 0 --steps 200000
```

With ε=0 the fixed policy is "optimal-path-then-camp": the agent walks S→G **once** and
then camps on the (non-absorbing) goal **forever**. So every non-goal state is visited
*exactly once ever* — one update, never revisited — while the goal is visited every step:

| state | V_learned | V^π (camp policy) |
|--:|--:|--:|
| 0 (S) | −0.41 | 150.2 |
| 14 | 11.4 | 200.0 |
| 15 (G) | **200.0** | **200.0** |

`V(goal) → goal_r/(1−γ) = 200` converges perfectly (visited forever); every other state is
stuck near its init 0 (`max|V−V^π| ≈ 189`) → check 2 **FAILS**. Check 1 still "passes" but
misleadingly: ~all steps are the zero-error goal self-loop, so the mean TD error → 0 even
though the start region was never learned. This is exactly the regular_vs_continuous
conclusion — *"the TD machinery is fine; it's a coverage problem"* — and is why the
`--env continuous` defaults turn on exploration (ε=0.4) and step-size decay.

## Grid-distance-aware overlap on a larger (8×8) grid — does *spatial* overlap help?

The `--encoder rdse` result above is overlap that is *misaligned* with value: it encodes
the **state index** as a scalar, so it shares bits between index-adjacent states (4↔5),
which on FrozenLake are value-distinct (path vs hole). That left an obvious question open:
**what if the overlap is *spatially* meaningful** — two tiles share *more* bits the
*closer* they are on the 2-D grid (small diagonal distance)? Spatial proximity is at least
*loosely* aligned with value (adjacent tiles often have similar returns), so grid-aware
overlap *might* generalize rather than interfere.

This experiment tests exactly that, with three changes the question needs:

1. **A bigger env** — the standard gym **8×8** FrozenLake (64 states, `grid_env.py`), so
   there is real room between tiles for a distance gradient to matter (the 4×4 is too
   small for "far" to differ from "near"). Optimal policy/path and the analytic V^π are
   derived generically by value iteration / policy evaluation.
2. **A grid-distance encoder** — `CoordinateEncoder` (`coordinate_encoder.py`): encode the
   tile's **(row, col)** with a real project encoder *per axis* and concatenate, so code
   overlap falls off with grid distance. Two backends, as asked: a contiguous **`scalar`**
   encoder and an **`rdse`** encoder.
3. **High dimensionality** — **1024-bit** SDRs (512 per axis).

For contrast we run the non-overlapping **`unique`** baseline and the value-misaligned
**`index-rdse`** (the 4×4 FAIL case) at the same 1024 bits. Everything is scored on the
same value checks against the exact analytic V^π, plus an **overlap-vs-distance**
diagnostic. (`run_overlap_experiment.py`.)

```bash
uv run python scripts/sparse_repr_td-learning/run_overlap_experiment.py
```
Outputs `results_overlap_8x8.json` / `sparse_repr_overlap_8x8.png`.

### How to read the metrics in the tables below

Everything in the two tables is one of two kinds of measurement: **how the encoder is
built** (computed from the codes alone, before any learning) and **how well value was
learned** (the learned values vs the exact analytic answer). Here is each column.

**The quantity under test — V(s) vs V^π(s).** The whole experiment compares two value
functions over the 64 tiles:

- **V^π(s)** — the *ground-truth* value of tile `s`: the expected total γ-discounted future
  reward from `s` onward under the fixed policy. Computed **exactly** by solving the Bellman
  equations on the known 8×8 MDP (`grid_env.policy_evaluation`) — no learning, it is the
  "right answer".
- **V(s)** — the value the **TD learner** ends up with, read out of the SDR as the mean of
  its active neurons' learned values. This is what the encoder + TD update actually produce.

Because the TD math itself is already validated (the 4×4 sanity test above), any gap
between `V` and `V^π` is attributable to the **encoder** smearing value across tiles that
should be kept distinct. That gap is the error we tabulate.

**Value scale (so the numbers have units).** On this 8×8 map `V^π` runs from about **−11.85**
(dead tiles far from the goal, accumulating the −1 step penalty) to **+11.85** (next to the
+10 goal) — a span of ~24 reward units. So read errors relative to that: **0.20 ≈ exact**
(~1 % of scale), **1.5 ≈ a visible bias** (~6 %), **9 ≈ the value is simply wrong** (the
learner rates a true −11 tile as ≈ −1). The script's PASS threshold is `tol = 0.63` (5 % of
scale).

**`mean |V−V^π|`** — the *average* absolute value error across the evaluated tiles. This is
the robust, headline comparison number: "on a typical tile, how biased is the learned
value?" The encoder **ranking** lives here (lower = the code preserves value structure
better).

**`max |V−V^π|` (covered)** — the *single worst* tile's error. One outlier can dominate it,
which is exactly its job: it surfaces the worst-case failure mode. Here the large maxes
(≈ 9) led us to the row/column-bleeding artifact (see finding 3) — a thing the mean hides.

**"covered" / `≥200 visits`** — the error is computed only over tiles the policy visits
enough to learn (≥ 200 times across the 2M training steps). A TD learner *cannot* learn a
tile it sees a handful of times; on 8×8 the hole-walled top-right pocket gets < 70 visits
and stays near its init value of 0. That is a **coverage** limit of the policy/map (the same
effect the continuous section documents), not an encoder fault, so we exclude it — otherwise
it would add identical noise to every encoder's `max` and obscure the comparison.

**`signed resid →`** — the late-training mean of the *signed* TD error (the Bellman
residual). `→0` means the learner **converged** (its values stopped having a systematic
under/over-shoot). This is the convergence check; note it can be ~0 even when values are
biased, so it is necessary but not sufficient — that is why we also check `|V−V^π|`.

**`grad` (monotonic)** — whether learned `V` rises monotonically along the optimal path
S→G. This is the *policy-relevant* property: even if absolute values are biased, a monotone
gradient means the **ordering** of states toward the goal is preserved (so a greedy policy
would still climb toward reward).

**Encoder-structure columns** (the first table) describe the code itself:

- **`k` (active bits)** — how many of the 1024 bits are ON per state (its sparsity).
  e.g. `coord-scalar` k=292 (28 %, dense) vs `coord-rdse` k=47 (4.6 %, sparse).
- **`corr(overlap, Euclidean dist)`** — across all tile *pairs*, the correlation between
  *how many bits two tiles share* and *how far apart they sit on the grid*. A strong
  **negative** value (e.g. −0.76) is the goal: closer tiles share more bits → the code
  literally encodes diagonal distance.
- **`max-pair / mean overlap`** — the shared-bit counts that show the overlap magnitude and
  how it falls off with distance.

### The encoders *do* encode diagonal distance

Code overlap is now a monotone-decreasing function of grid (Chebyshev) distance, and
correlates with the diagonal (Euclidean) separation — the thing the index-scalar `rdse`
fails to do:

| encoder (1024-bit) | active bits *k* | corr(overlap, Euclidean dist) | mean overlap at dist 1 → 7 |
|---|--:|--:|---|
| `coord-scalar` | 292 (28%) | **−0.76** | 215 → 47 |
| `coord-rdse`   | 47 (4.6%) | **−0.62** | 30 → 7 |
| `index-rdse` (index as scalar) | 24 | −0.54¹ | (overlap follows row-major index, not 2-D) |
| `unique` (baseline) | 16 | 0 (disjoint) | 0 |

¹ `index-rdse` has *some* Euclidean correlation only because the row-major index loosely
tracks position; its overlap still couples spatially-distant tiles (e.g. end-of-row 7 with
start-of-row 8). Note also that the contiguous **scalar** encoder must spend **k=292**
active bits (28% dense) to overlap neighbours at 512 bits/axis, whereas **rdse** achieves
the same spatial gradient at **k=47** (4.6%) — *RDSE is the far more efficient grid
encoder* (its axis radius is independent of SDR width; the scalar encoder's is not).

### Result — grid-aware overlap beats index overlap, but still biases value

Episodic 8×8, 1024-bit, 400k steps × 5 seeds, ε=0.35, α=0.5 γ=0.95 λ=0.6, α-decay 5e-4.
Value error is measured over **adequately-visited** states (≥200 visits/2M steps): the
top-right F-pocket `{5,6,7,15}` is hole-walled and gets <70 visits, so it is
coverage-starved — the documented *coverage* limitation, not an encoder effect (see the
continuous section above). The next-covered state has >170 visits, so 200 cleanly excludes
the pocket.

| encoder | overlap | `mean\|V−V^π\|` | `max\|V−V^π\|` (covered) | signed resid → | grad |
|---|---|--:|--:|--:|:--:|
| `unique` (none) | disjoint | **0.20** | 0.65 | →0 | monotonic |
| `coord-rdse` | spatial (Euclid −0.62) | **1.52** | 9.1 | →0 | monotonic |
| `coord-scalar` | spatial (Euclid −0.76) | **1.64** | 9.2 | →0 | monotonic |
| `index-rdse` | index-adjacency | **2.07** | 9.4 | →0 | monotonic |

Reading this:

- **Spatial overlap is *less harmful* than index overlap.** Both `coord-*` encoders beat
  `index-rdse` on mean value error (1.5–1.6 vs 2.1) — aligning the code's similarity with
  the grid's geometry (which loosely tracks value) interferes less than aligning it with
  the meaningless row-major index. So *yes*, the **kind** of overlap matters, exactly as
  the 4×4 section predicted.
- **But it is still biased vs no overlap** (`unique` ≈ 0.20). On FrozenLake even spatial
  neighbours can be value-distinct (a hole abuts a path tile), so *any* overlap couples
  some value-distinct states. The thesis TD update remains **unbiased only on a code that
  separates value-distinct states** — the central finding, now confirmed for *spatial*
  overlap too.
- **The worst errors expose a structural artifact of *concatenated coordinate* codes.** The
  high-error covered states are the bottom-left corner `{56,57,58}=(7,0..2)` — well-visited
  (>1100 visits) yet learned as ≈ −1 when V^π ≈ −11. Cause: concatenating per-axis codes
  means **any two tiles in the same row (or column) share that entire axis half** (~all
  512 row-bits), *regardless of how far apart they are along the row*. Row 7 mixes the
  goal-region tiles (high V) with the dead bottom-left corner (low V), so value **bleeds
  along the row**. The code is effectively **cross-shaped, not radial** — it captures
  *per-axis* (Manhattan-ish) proximity but over-couples whole rows/columns, which is why
  corr(overlap, Euclidean) tops out at −0.76, not −1. A genuinely *radial* 2-D
  receptive-field (grid-cell) encoder would avoid the cross-coupling, but the per-axis
  scalar/RDSE construction the experiment was asked to use cannot.
- **The TD *principle* holds throughout.** For every encoder the signed TD residual → 0
  (converges) and V rises monotonically along the optimal path (the policy-relevant
  *ordering* survives) — only the absolute values are biased by feature overlap, the same
  conclusion as the 4×4 `rdse` case.

**Overlap radius is a soft knob, not a cliff.** Sweeping the coordinate encoder's overlap
radius (`--radius-tiles`, RDSE backend, k fixed at 47, 200k×5) barely moves accuracy —
mean `|V−V^π|` = 1.55 / 1.54 / 1.45 / 1.50 at radius = 0.5 / 1 / 2 / 3 tiles — so the
result is robust to *how much* spatial overlap you dial in; what matters is that the
overlap is spatial at all (and that the code still separates value-distinct tiles).

```bash
# reproduce the radius sweep
for r in 0.5 1.0 2.0 3.0; do
  uv run python scripts/sparse_repr_td-learning/run_overlap_experiment.py \
      --encoders coord-rdse --radius-tiles $r --steps 200000 --no-plot
done
```

**Takeaway.** Making SDR overlap track the tiles' **diagonal grid distance** is the
*right kind* of overlap — it generalizes (beats index-overlap) and preserves the value
ordering — but on FrozenLake it does not recover exact values, because (a) spatial
adjacency is still not perfectly value-aligned (holes), and (b) the *concatenated*
scalar/RDSE coordinate code couples entire rows/columns. The thesis TD math is, again,
not the limiting factor: it is sound wherever the representation keeps value-distinct
states apart.

## Control: does the thesis action-selection solve the task?

The sanity test above only checks the *value math*. The next step is the thesis **control
logic** — the D1/D2 Go/No-Go reward circuitry and voluntary action (§8–9) — built on the
same sparse substrate, to see if the agent can actually **navigate to the goal**.

`sparse_control.py` (`GoNoGoActor`) + `run_control.py`:

- **D1 (Go) / D2 (No-Go)** — two per-neuron value populations over the SDR, shared
  replacing traces. **The B7 fix**: the TD error is sign-split — δ≥0 updates Go, δ<0
  updates No-Go. The combined value `V = V_go − V_nogo` then evolves *exactly* as a
  standard TD(λ) critic (ΔV = α·δ·trace in both branches), but Go and No-Go are now
  **distinct populations** (measured mean `|V_go − V_nogo| ≈ 4.5 > 0`). In the full HTM
  agent B7 was the blocker — D1/D2 learned identical values, so the Go−NoGo motor signal
  was zero everywhere and the policy never moved.
- **Learned one-step model** `M[sdr][a] = (next_sdr, reward, done)` — the functional
  stand-in for the L4/L2/L5 distal Temporal Memory prediction, built from experience
  (your chosen "learn it online" option). Unknown `(s,a)` → no voluntary drive → base
  excitation explores it.
- **Voluntary action (§9)** — a next state that is both *distally predicted* (model-known)
  and *apically depolarized* (Go/No-Go value) drives the motor neurons. Net motor drive
  `drive(a) = [r + γ·V_go(s')] − [γ·V_nogo(s')] = r + γ·V(s') = Q(s,a)`. (The immediate
  reward is in the lookahead because the *episodic* goal is terminal with `V(goal)=0`;
  without it the agent would never step onto the goal.)
- **Motor (§6 / §9.2)** — n_actions×3 motor neurons, `drive + base random excitation`,
  **top-3 WTA**, action with the most active neurons wins. Base excitation = exploration:
  random when values ≈ 0, increasingly greedy as the Go/No-Go values grow.

### Result — SOLVED (episodic)

Episodic FrozenLake, 3000 episodes × 8 seeds, α=0.5 γ=0.95 λ=0.6, base excitation 4.0:

| encoder | greedy success → goal | greedy steps | `mean\|V_go−V_nogo\|` (B7) | solved |
|---|--:|--:|--:|:--:|
| `unique`   | **100%** (±0%) | 6.00 (optimal) | 4.55 | ✅ |
| `category` | **100%** (±0%) | 6.00 (optimal) | 4.55 | ✅ |
| `rdse`     | **100%** (±0%) | 6.00 (optimal) | 5.89 | ✅ |

Success rate reaches 1.0 within ~50 episodes and the greedy policy is exactly optimal
(6 steps S→G). The learned `V = V_go − V_nogo` rises along the path toward the goal (a
small dip at state 13 is an on-policy artifact — it neighbours hole 12, so exploration
there raises V_nogo; it doesn't affect the optimal greedy policy). Plot:
`sparse_control.png`.

Notably **RDSE solves it too**, despite the value *bias* its overlap caused in the value
sanity test: control only needs the action *ranking* to be correct, and on this task the
overlap doesn't flip the argmax.

### Result — SOLVED (continuous, the thesis default)

The thesis's actual run config is **reset-free / continuous** (the goal is a non-absorbing
attractor). Running the same control agent on the real `ContinuousFrozenLake`
(`--env continuous`) surfaces the exploration/coverage problem from
`regular_vs_continuous_fzlake/`, and the fix is the same — **strong, annealed exploration**:

| continuous setting (400k steps × 8 seeds) | greedy success from S | V(S) | what happens |
|---|--:|--:|---|
| base excitation only (no ε) | 25% (±43%) | −4.6 | camps at goal → **start-region starvation** |
| crank base excitation (100–500) | 0–25% | ≤ −47 | near-random → learns a *random walk's* value (all negative) |
| **ε = 0.6 annealed → 0.05 (GLIE)** | **100% (±0%)** | **+45** | **SOLVED, 6 steps (optimal)** |

Why each rung:

- **No/low exploration → camps.** Once `V(goal)≈200`, the value-greedy agent sits on the
  goal; with no resets the random-walk excursions rarely reach all the way back to S, so
  the start region never converges (`V(S)` negative, greedy from S fails). This is exactly
  the coverage starvation the value-only continuous experiment showed at ε=0.
- **Cranking base excitation doesn't help** — it's a *magnitude* on a ~200-scale attractor,
  so any value that randomises the small-valued start region also makes the whole policy
  near-random, which just learns the (useless, all-negative) value of a random walk.
- **The fix is an explicit exploration *rate*, GLIE-annealed.** Added `epsilon` to
  `GoNoGoActor` (the controllable form of "base excitation produces random motion") and
  anneal it ε 0.6→0.05 over the run. High ε early gives whole-grid coverage (so the start
  region converges); annealing it out lets the on-policy value settle toward the greedy
  policy so the argmax stops flipping. Result: **100% greedy success, 6 steps (optimal)**,
  monotonic `V` up to `V(goal)≈186`, B7 gap `|V_go−V_nogo|≈117`. Plot:
  `sparse_control_continuous.png` (greedy success snaps to 1.0 as ε anneals out).

This reproduces the `regular_vs_continuous_fzlake/` conclusion at the level of the thesis
control logic: the machinery is sound; the reset-free goal-attractor is what demands the
strong exploration (ε≈0.6) — a property of the *task*, not the agent.

### What this demonstrates about the full HTM agent

`tasks/todo.md` diagnosed that the full agent's failure was **not** in the B1–B6 value
math but in **B7** (D1/D2 collapsing to identical values) plus the Section-C action
pathway. This reconstruction confirms that diagnosis end-to-end: with the value math
intact (B1–B6, validated above), the *only* change needed to get goal-directed behaviour
is the **B7 sign-split** so Go≠No-Go, wired through a voluntary-action/motor stage. Once
that is in place, the thesis control architecture solves the environment.

## Bottom line

The thesis TD(λ) reward-circuitry update (Eqs 5.1–5.5) is a sound, unbiased
value-learner on a sparse distributed code: it converges to the exact Bellman values
and builds the correct discounted gradient toward reward. And with the **B7 sign-split**
restored, the thesis **control logic** (D1/D2 Go/No-Go + voluntary action) built on that
substrate **solves episodic FrozenLake at optimal 100% success**. The pieces that stop
the *full HTM* agent are therefore B7 + the Section-C action pathway — not the TD value
math, and not the control architecture itself, both validated here.
