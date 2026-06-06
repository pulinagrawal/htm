# Regular vs. Continuous FrozenLake under basic TD-learning

Does a basic temporal-difference setup (tabular **SARSA**) behave the same on the
**continuous** FrozenLake (`src/continuous_frozen_lake.py::ContinuousFrozenLake`, no
episode boundaries) as on the **regular** episodic gym `FrozenLake-v1`?

Short answer: **the TD update itself is sound in both** (errors decay, a correct Bellman
value gradient forms near the goal), but the continuous, reset-free setup changes the
*task* and the *exploration dynamics* enough that a naive port of episodic SARSA
hyperparameters learns a much worse policy from the start state.

## What is held identical

Both conditions use the **same** 4×4 grid, **same** deterministic dynamics (gym's default
4×4 map is byte-for-byte the `TILES` map in `ContinuousFrozenLake`), **same** reward
schedule `(goal=+10, hole=−10, step=−1)`, **same** SARSA update, and **same**
hyperparameters. The *only* difference:

| | Regular (`EpisodicFrozenLake`) | Continuous (`ContinuousFrozenLake`) |
|---|---|---|
| Goal / hole | terminal | just another tile — agent keeps going |
| TD target at goal/hole | `r` (bootstrap cut) | `r + γ·Q(s′,a′)` (always bootstraps) |
| Reset | every episode → back to S | never |
| Step budget | matched on **total env steps**, not episodes | — |

## Files

- `td_learning.py` — shared tabular SARSA core (the only thing the two loops do
  differently is pass `done=True` to cut the bootstrap at a terminal).
- `environments.py` — `EpisodicFrozenLake` (gym, rewards remapped by tile to match the
  continuous schedule) + a factory that builds the real `ContinuousFrozenLake`.
- `run_comparison.py` — trains both over N seeds, evaluates, plots, prints a summary,
  writes `results.json`.

## Run

```bash
uv run python scripts/regular_vs_continuous_fzlake/run_comparison.py            # defaults: 30 runs, 30k steps
uv run python scripts/regular_vs_continuous_fzlake/run_comparison.py --runs 30 --steps 30000 --epsilon 0.6
```

Outputs `regular_vs_continuous_td.png` and `results.json` in this folder.

## What we measure ("do TD principles hold?")

1. **Policy correctness** — greedy rollout from S, evaluated in a *common* episodic test
   env, must reach the goal.
2. **Convergence** — mean `|TD error|` decays toward 0.
3. **Bellman value gradient** — `V(s)=max_a Q(s,a)` rises monotonically along the optimal
   path `0→4→8→9→13→14→15`.

## Findings (30 runs × 30k steps, α=0.1, γ=0.95, ε=0.1)

| metric | episodic | continuous |
|---|--:|--:|
| Greedy success rate (→ goal) | 96.7% | 18.4% |
| Mean steps-to-goal at eval | 6.00 (optimal) | 9.53 |
| Steps until first goal visit | 131 | 39 |
| Final mean \|TD error\| (last 1k) | 0.77 | 2.76 |
| V(start) | +1.43 | −0.28 |
| V(goal) | 0.00 (terminal) | +180.3 |

Three things stand out, and all three are *expected consequences of removing episodes*,
not bugs:

1. **The goal is non-absorbing → it becomes an attractor.** From state 15 the actions
   `Down`/`Right` keep the agent *on* the goal, banking `+10` every step. TD correctly
   drives `V(goal) → goal_r/(1−γ) = 200`; we measure ≈180. The "cumulative goal visits"
   curve makes this concrete: the continuous agent visits the goal ~28k times in 30k steps
   (it camps there), versus ~4k for the episodic agent (once per episode).

2. **No resets → exploration starvation near the start.** Because the agent camps near the
   high-value goal, the ε=0.1 behavior policy almost never wanders the ~6 steps back to S
   against the value gradient. So start-region Q-values stay near their initial 0
   (`V(start) ≈ −0.28`, unconverged), and the greedy policy from S is poor (18% success).
   The episodic agent gets start-region coverage *for free* from every reset.

3. **The TD machinery is fine — it's a coverage problem.** Raising exploration rescues the
   continuous agent completely:

   | ε | continuous success | V(start) |
   |--:|--:|--:|
   | 0.1 | 33% | −0.26 (unconverged) |
   | 0.3 | 50% | −0.55 |
   | 0.6 | **100%** | **+6.32 (converged, positive)** |

   At ε=0.6 the continuous agent reaches 100% greedy success and V(start) goes positive —
   TD-learning *does* hold for `ContinuousFrozenLake`; it just needs enough state coverage
   to compensate for the lack of resets.

## Caveats when reading the numbers

- **`|TD error|` magnitudes are not directly comparable** across conditions: continuous
  values are ~100× larger (O(100) vs O(10)), so equal *relative* convergence shows up as
  larger *absolute* error. Both curves decay; the continuous one plateaus higher partly
  for this scale reason and partly because the under-visited start keeps producing error.
- **SARSA is on-policy**, so it learns the value of the ε-greedy policy, not the optimal
  one. That is why episodic `V(start)` can go *negative* at high ε even though the greedy
  policy extracted from it still solves the maze — the values reflect the noisy behavior
  policy, the argmax does not.
- The "V monotonic along optimal path" fraction is low for episodic because `V(goal)=0`
  (terminal) breaks the final step of the gradient by construction — read it together with
  the value-gradient plot rather than alone.

## Bottom line

Basic TD-learning principles **transfer** to `ContinuousFrozenLake`: the update converges
where the agent actually visits, and it builds the correct discounted Bellman gradient
toward the goal. But dropping episode boundaries (a) turns the goal into a non-absorbing
attractor that inflates values and induces camping, and (b) removes the free exploration
that resets provide — so the continuous setup needs substantially more exploration to learn
a good policy from the start state.
