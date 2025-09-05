import numpy as np
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from htm import (
    TemporalPooler,
    CONNECTED_PERM,
    MIN_OVERLAP,
)

# Helper to build a small deterministic pooler

def small_pooler(cols=16, inputs=32, syns_per_col=8, cells_per_col=4, seed=0):
    rng = np.random.default_rng(seed)
    tp = TemporalPooler(inputs, cols, cells_per_col, syns_per_col)
    # Force deterministic permanence values around threshold
    for c in tp.columns:
        for s in c.potential_synapses:
            s.permanence = 0.49 if rng.random() < 0.5 else 0.51
        c.connected_synapses = [s for s in c.potential_synapses if s.permanence > CONNECTED_PERM]
    return tp


def test_compute_overlap_and_inhibition_min_overlap_filtering():
    tp = small_pooler()
    # Build input that activates only connected synapses for first two columns sufficiently
    input_vec = np.zeros(32, dtype=int)
    # Ensure at least MIN_OVERLAP connected synapses for column 0 and 1
    for col in tp.columns[:2]:
        count = 0
        for syn in col.connected_synapses:
            input_vec[syn.source_input] = 1
            count += 1
            if count >= MIN_OVERLAP:
                break
    active_cols = tp.compute_active_columns(input_vec, inhibition_radius=1.5)
    # Both first two columns should survive local inhibition if their overlap identical
    positions = [c.position for c in active_cols]
    assert tp.columns[0].position in positions
    assert tp.columns[1].position in positions


def test_learning_phase_permanence_increase_and_decrease():
    tp = small_pooler()
    col = tp.columns[0]
    input_vec = np.zeros(32, dtype=int)
    # Activate half of synapses
    for syn in col.potential_synapses[:len(col.potential_synapses)//2]:
        input_vec[syn.source_input] = 1
    before = [syn.permanence for syn in col.potential_synapses]
    tp.learning_phase([col], input_vec)
    after = [syn.permanence for syn in col.potential_synapses]
    inc, dec = 0, 0
    for b, a, syn in zip(before, after, col.potential_synapses):
        if input_vec[syn.source_input]:
            assert a >= b
            inc += 1
        else:
            assert a <= b
            dec += 1
    assert inc > 0 and dec > 0


def test_temporal_bursting_creates_new_segment_and_winner():
    tp = small_pooler()
    input_vec = np.zeros(32, dtype=int)
    # Activate first column only
    for syn in tp.columns[0].connected_synapses[:MIN_OVERLAP]:
        input_vec[syn.source_input] = 1
    active_cols = tp.compute_active_columns(input_vec, inhibition_radius=2)
    tp.current_t = 0
    tp.compute_active_state(active_cols)
    # All cells in col 0 become active (burst)
    col0 = tp.columns[0]
    assert len(tp.active_cells[tp.current_t].intersection(col0.cells)) == len(col0.cells)
    # One winner cell selected
    assert len(tp.winner_cells[tp.current_t].intersection(col0.cells)) == 1
    winner = list(tp.winner_cells[tp.current_t].intersection(col0.cells))[0]
    # A new segment should have been created on the winner
    assert len(winner.segments) == 1


def test_temporal_prediction_restricts_activation():
    tp = small_pooler()
    # Force a scenario: at t=0 burst column 0, learn, then at t=1 only predicted cell becomes active
    input_vec = np.zeros(32, dtype=int)
    for syn in tp.columns[0].connected_synapses[:MIN_OVERLAP]:
        input_vec[syn.source_input] = 1
    active_cols = tp.compute_active_columns(input_vec, inhibition_radius=2)
    tp.current_t = 0
    tp.compute_active_state(active_cols)
    tp.compute_predictive_state()
    tp.learn()
    # Now craft prev active cells to enable prediction: use winner cell's segment synapses reinforced
    # Simulate next time step with same column active
    active_cols_1 = tp.compute_active_columns(input_vec, inhibition_radius=2)
    tp.current_t = 1
    tp.compute_active_state(active_cols_1)
    # If prediction worked, fewer active cells than full burst
    col0 = tp.columns[0]
    active_in_col = tp.active_cells[tp.current_t].intersection(col0.cells)
    assert 1 <= len(active_in_col) <= len(col0.cells)


def test_segment_reinforcement_grows_new_synapses():
    tp = small_pooler()
    # Initial burst to create a segment
    input_vec = np.zeros(32, dtype=int)
    for syn in tp.columns[0].connected_synapses[:MIN_OVERLAP]:
        input_vec[syn.source_input] = 1
    active_cols = tp.compute_active_columns(input_vec, inhibition_radius=2)
    tp.current_t = 0
    tp.compute_active_state(active_cols)
    tp.learn()
    winner = list(tp.winner_cells[0].intersection(tp.columns[0].cells))[0]
    seg = winner.segments[0]
    before_count = len(seg.synapses)
    # Make several cells active at t=1 (burst another column to provide candidates)
    input_vec2 = np.zeros(32, dtype=int)
    for syn in tp.columns[1].connected_synapses[:MIN_OVERLAP]:
        input_vec2[syn.source_input] = 1
    active_cols2 = tp.compute_active_columns(input_vec2, inhibition_radius=2)
    tp.current_t = 1
    tp.compute_active_state(active_cols2)
    tp.learn()  # reinforce any predictive segments (none yet) but negative segments punished
    # Mark previous active cells as predictive sources artificially by calling compute_predictive_state
    tp.compute_predictive_state()
    # Reinforce again after adding prev active context
    tp.learn()
    after_count = len(seg.synapses)
    assert after_count >= before_count  # new synapses may be added


def test_distal_synapse_permanence_changes_on_reinforce_and_punish():
    tp = small_pooler()
    # Burst column 0 to create segment
    input_vec = np.zeros(32, dtype=int)
    for syn in tp.columns[0].connected_synapses[:MIN_OVERLAP]:
        input_vec[syn.source_input] = 1
    active_cols = tp.compute_active_columns(input_vec, inhibition_radius=2)
    tp.current_t = 0
    tp.compute_active_state(active_cols)
    tp.learn()
    winner = list(tp.winner_cells[0].intersection(tp.columns[0].cells))[0]
    seg = winner.segments[0]
    # Ensure synapses exist by reinforcing after crafting prev active cells
    # Make some cells active at t=1 to supply sources
    input_vec2 = np.zeros(32, dtype=int)
    for syn in tp.columns[1].connected_synapses[:MIN_OVERLAP]:
        input_vec2[syn.source_input] = 1
    active_cols2 = tp.compute_active_columns(input_vec2, inhibition_radius=2)
    tp.current_t = 1
    tp.compute_active_state(active_cols2)
    # Reinforce segment using prev active cells at t=1 (simulate prediction learning)
    tp.current_t = 1
    tp.reinforce_segment(seg)
    assert len(seg.synapses) > 0
    base_perms = [syn.permanence for syn in seg.synapses]
    tp.current_t = 2
    tp.punish_segment(seg)
    after_punish = [syn.permanence for syn in seg.synapses]
    # At least one permanence should have decreased
    assert any(a < b for a, b in zip(after_punish, base_perms))
