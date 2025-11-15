import sys
import pathlib
import numpy as np
import random

# Ensure module import
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from htm import TemporalPooler  # noqa: E402

def test_field_specific_prediction_filtering():
    random.seed(42)
    np.random.seed(42)
    # Two coding fields: lengths 10 and 20
    field1 = np.random.randint(0, 2, size=10)
    field2 = np.random.randint(0, 2, size=20)
    total = field1.size + field2.size

    tp = TemporalPooler(input_space_size=total, column_count=64, cells_per_column=4, initial_synapses_per_column=12)

    # Run one spatial step with dict input to establish field metadata & mapping
    tp.current_t = 0
    tp.run({"f1": field1, "f2": field2}, mode="spatial", inhibition_radius=2)

    # Sanity: field metadata captured
    assert tp.field_ranges == {"f1": (0, 10), "f2": (10, 30)}
    assert tp.column_field_map, "column_field_map should be populated after spatial run with dict input"

    # Manually craft a predictive set using only columns dominated by field f1
    f1_columns = [col for col, fname in tp.column_field_map.items() if fname == "f1"]
    # Choose up to 5 columns for prediction
    chosen_cols = f1_columns[:5]
    predictive_cells = set()
    for col in chosen_cols:
        predictive_cells.add(col.cells[0])  # pick first cell of each column
    tp.predictive_cells[0] = predictive_cells

    # Filter predictions by field
    predicted_f1_mask = tp.get_predictive_columns(t=0, field_name="f1")
    predicted_f2_mask = tp.get_predictive_columns(t=0, field_name="f2")
    f1_idx_set = {i for i, v in enumerate(predicted_f1_mask) if v}
    f2_idx_set = {i for i, v in enumerate(predicted_f2_mask) if v}
    chosen_idx_set = {tp.columns.index(c) for c in chosen_cols}
    assert f1_idx_set == chosen_idx_set, "Filtered predictive columns for field f1 should match manually set columns"
    assert not f2_idx_set, "No predictive columns should be returned for field f2 when only f1 cells were marked predictive"
