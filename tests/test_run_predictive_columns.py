import sys
import pathlib
import numpy as np
from typing import Set

# Ensure module import path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from htm import TemporalPooler, CONNECTED_PERM, SEGMENT_ACTIVATION_THRESHOLD, DistalSynapse, Segment  # noqa: E402


def test_run_produces_predictive_columns_accessible_via_api():
    """Integration test: using TemporalPooler.run() (spatial mode) computes predictive
    cells and these are exposed correctly through get_predictive_columns.

    We manually pre-create a distal segment on a cell whose synapses reference
    other cells that will be active in the same bursting timestep so that the
    segment is immediately active and the owning cell becomes predictive in
    that first run() call.
    """
    inputs = 32
    cols = 16
    cells_per_col = 6
    syns_per_col = 8
    tp = TemporalPooler(inputs, cols, cells_per_col, syns_per_col)

    # Choose first column; build an input vector to ensure it becomes active.
    col0 = tp.columns[0]
    input_vec = np.zeros(inputs, dtype=int)
    # Activate enough connected synapse sources for overlap >= MIN_OVERLAP.
    for syn in col0.connected_synapses[:3]:  # 3 is MIN_OVERLAP in module
        input_vec[syn.source_input] = 1

    # Manually attach an immediately activatable segment to cell0.cells[0].
    target_cell = col0.cells[0]
    if not target_cell.segments:
        seg = Segment()
        # Add synapses pointing to other cells in the same column that will burst active.
        # Use exactly SEGMENT_ACTIVATION_THRESHOLD sources so the segment becomes active.
        sources = col0.cells[1 : 1 + SEGMENT_ACTIVATION_THRESHOLD]
        for src in sources:
            seg.synapses.append(DistalSynapse(src, CONNECTED_PERM + 0.1))
        target_cell.segments.append(seg)

    tp.current_t = 0
    # Single spatial run (no time advance needed for this assertion)
    result = tp.run(input_vec, mode="spatial", inhibition_radius=2)
    predictive_cells_vec = result["predictive_cells"]  # numpy array
    # Convert vector back to set of cells for assertion convenience
    pred_cells: Set = set()
    for col_idx, col in enumerate(tp.columns):
        base = col_idx * tp.cells_per_column
        for local_idx, cell in enumerate(col.cells):
            if predictive_cells_vec[base + local_idx]:
                pred_cells.add(cell)
    assert target_cell in pred_cells, "Expected manually primed cell to be predictive"

    # API: get predictive columns (explicit t and default) should both include col0
    pred_cols_explicit = tp.get_predictive_columns(t=0)
    pred_cols_default = tp.get_predictive_columns()
    # Convert binary vectors to sets of column indices
    explicit_idxs = {i for i, v in enumerate(pred_cols_explicit) if v}
    default_idxs = {i for i, v in enumerate(pred_cols_default) if v}
    assert 0 in explicit_idxs, "Column owning predictive cell should be returned by get_predictive_columns(t=0)"
    assert explicit_idxs == default_idxs, "Default predictive column retrieval should target latest timestep"

    # Sanity: each predictive column must contain at least one predictive cell
    for i, v in enumerate(pred_cols_default):
        if v:
            col = tp.columns[i]
            assert any(cell in pred_cells for cell in col.cells), "Predictive column lacks corresponding predictive cell"
