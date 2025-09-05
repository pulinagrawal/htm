import sys
import pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from htm import TemporalPooler


def test_dict_field_prediction_filtering():
    # Two fields of different sizes
    fields = {
        'vision': np.random.randint(0, 2, size=30),
        'audio': np.random.randint(0, 2, size=20)
    }
    total_inputs = sum(v.shape[0] for v in fields.values())
    tp = TemporalPooler(total_inputs, column_count=64, cells_per_column=4, initial_synapses_per_column=12)

    # Time 0
    active_cols = tp.compute_active_columns(fields, inhibition_radius=2)
    tp.current_t = 0
    tp.compute_active_state(active_cols)
    tp.compute_predictive_state()

    # Basic retrieval default (latest)
    all_pred_cols = tp.get_predictive_columns()
    assert isinstance(all_pred_cols, set)

    # Field-specific retrieval
    vision_preds = tp.get_predictive_columns(field_name='vision')
    audio_preds = tp.get_predictive_columns(field_name='audio')
    # Sets should be subsets of all
    assert vision_preds.issubset(all_pred_cols)
    assert audio_preds.issubset(all_pred_cols)

    # -1 (previous) should work (may be empty if only one timestep)
    prev_preds = tp.get_predictive_columns(t=-1)
    assert isinstance(prev_preds, set)
