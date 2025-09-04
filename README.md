# HTM Prototype

This repository contains a prototype implementation of a combined Spatial Pooler and (simplified) Temporal Memory (named `TemporalPooler` in code) based loosely on the BAMI (Biological And Machine Intelligence) book pseudocode. The implementation is incomplete and diverges from canonical BAMI algorithms in several key ways (see below).

## Key Deviations from BAMI Pseudocode

1. Distal Synapses Representation: Synapses in temporal segments reference raw input indices (`source_input`) rather than prior active cells. BAMI TM uses cell-to-cell distal synapses.
2. Active State Computation: All cells in an active column are marked active (`bursting`) unconditionally; proper predictive cell selection vs. bursting is not implemented.
3. Segment / Synapse Data Structures: Each cell is initialized with exactly one segment; creation of new segments and synapses during learning is not implemented.
4. Learning Thresholds: `THRESHOLD` is a constant applied uniformly; separate activation and learning thresholds are not differentiated for temporal segments.
5. Inhibition / Local Activity: `kth_score` assumes at least `DESIRED_LOCAL_ACTIVITY` neighbors and may raise if not; no guard for small neighborhoods.
6. Duty Cycle / Boosting: Placeholders for boosting variables exist but boosting adaptation logic (increase boost when below min duty cycle) is not implemented.
7. Overlap Calculation: Uses a simple count of connected synapses with active input bits; no synapse permanence noise or tie-breaking logic.
8. Segment Update Logic: `update_segment_list` replaces synapse list wholesale; BAMI appends new synapses and tracks learning segments.

## Tests

A pytest suite (`tests/`) verifies implemented behaviors and marks divergences using `xfail`.

Run tests:
```bash
python -m pip install -r requirements.txt
pytest -q
```

## Next Steps

- Implement proper temporal memory distal synapses between cells.
- Add predictive state logic with correct bursting and segment matching.
- Add creation of new segments and synapses per BAMI learning rules.
- Implement duty cycle tracking and boosting adaptations.
- Harden inhibition for edge neighborhoods.
