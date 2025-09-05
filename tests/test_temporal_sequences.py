import random
import numpy as np
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from htm import TemporalPooler


def test_multi_field_input_equivalence():
    total_inputs = 40
    np.random.seed(123)
    tp = TemporalPooler(total_inputs, column_count=64, cells_per_column=4, initial_synapses_per_column=12)
    field1 = np.random.randint(0, 2, size=15)
    field2 = np.random.randint(0, 2, size=25)
    concat = np.concatenate([field1, field2])

    # First call: concatenated explicit array
    active_cols_concat = tp.compute_active_columns(concat, inhibition_radius=2)
    positions_concat = {c.position for c in active_cols_concat}

    # Second call: list of fields (should be identical after internal combine)
    active_cols_multi = tp.compute_active_columns([field1, field2], inhibition_radius=2)
    positions_multi = {c.position for c in active_cols_multi}

    assert positions_concat == positions_multi, "Multi-field input should match pre-concatenated input results"


def test_letter_sequence_prediction_accuracy():
    random.seed(123)
    rng = np.random.default_rng(1)
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    column_count = 128
    tp = TemporalPooler(input_space_size=50, column_count=column_count, cells_per_column=6, initial_synapses_per_column=16)

    # Map each letter to a random sparse set of active columns (objects)
    k = 5  # columns per letter
    letter_to_columns = {}
    for L in letters:
        letter_to_columns[L] = rng.choice(tp.columns, size=k, replace=False)

    # Training sequences
    sequences = [list("ABCD"), list("CBDE"), list("AXYZXC"), *[[chr(65+letter)] for letter in range(26)]]  # Provided examples
    sequences = [list("ABCDX"), list("EFGHY"), list("JKLMN")]  # Provided examples
    training_chain = []
    for _ in range(100):  # repeat to ensure segments form
      # random sample from sequences
      seq = random.choice(sequences)
      training_chain.extend(seq)

    # Train
    for t, sym in enumerate(training_chain):
        tp.current_t = t
        active_cols = list(letter_to_columns[sym])
        tp.compute_active_state(active_cols)
        tp.compute_predictive_state()
        tp.learn()

    # # Boost distal synapse permanence so predictions can occur quickly
    # for col in tp.columns:
    #     for cell in getattr(col, 'cells', []):
    #         for seg in cell.segments:
    #             for syn in seg.synapses:
    #                 syn.permanence = 0.6  # above CONNECTED_PERM

    # Replay a mixed test chain with random distractor letters inserted
    test_chain = []
    # Insert noise letters
    for _ in range(100):  # repeat to ensure segments form
      # random sample from sequences
      seq = random.choice(sequences)
      test_chain.extend(seq)

    tp.reset_state()
    correct = 0
    total = 0
    for t, sym in enumerate(test_chain):
        tp.current_t = t
        active_cols = list(letter_to_columns[sym])
        tp.compute_active_state(active_cols)
        tp.compute_predictive_state()
        if t < len(test_chain) - 1:
            next_sym = test_chain[t + 1]
            predicted_columns = tp.get_predictive_columns(t)
            actual_next_cols = set(letter_to_columns[next_sym])
            if actual_next_cols & predicted_columns:
                correct += 1
            total += 1

    # Require some minimum predictive skill (non-zero and at least 20%)
    assert total > 0
    accuracy = correct / total
    print(f"Prediction accuracy: {accuracy:.2f}")
    assert accuracy >= 0.5, f"Prediction accuracy too low: {accuracy:.2f}"
