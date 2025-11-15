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
    active_mask = tp.compute_active_columns(fields, inhibition_radius=2)
    active_cols = [tp.columns[i] for i, v in enumerate(active_mask) if v]
    tp.current_t = 0
    tp.compute_active_state(active_cols)
    tp.compute_predictive_state()

    # Basic retrieval default (latest)
    all_pred_mask = tp.get_predictive_columns()
    assert all_pred_mask.ndim == 1

    # Field-specific retrieval
    vision_mask = tp.get_predictive_columns(field_name='vision')
    audio_mask = tp.get_predictive_columns(field_name='audio')
    # Subset property: indices set containment
    all_set = {i for i, v in enumerate(all_pred_mask) if v}
    vis_set = {i for i, v in enumerate(vision_mask) if v}
    aud_set = {i for i, v in enumerate(audio_mask) if v}
    assert vis_set.issubset(all_set)
    assert aud_set.issubset(all_set)

    # -1 (previous) should work (may be empty if only one timestep)
    prev_mask = tp.get_predictive_columns(t=-1)
    assert prev_mask.ndim == 1
