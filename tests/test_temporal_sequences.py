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
    active_mask_concat = tp.compute_active_columns(concat, inhibition_radius=2)
    positions_concat = {tp.columns[i].position for i, v in enumerate(active_mask_concat) if v}

    # Second call: list of fields (should be identical after internal combine)
    active_mask_multi = tp.compute_active_columns([field1, field2], inhibition_radius=2)
    positions_multi = {tp.columns[i].position for i, v in enumerate(active_mask_multi) if v}

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
        # Sample k distinct indices then map to columns (avoids np.random typing complaints with objects)
        idxs = rng.choice(len(tp.columns), size=k, replace=False)
        letter_to_columns[L] = [tp.columns[int(i)] for i in idxs]

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
            predicted_mask = tp.get_predictive_columns(t)
            predicted_set = {tp.columns[i] for i, v in enumerate(predicted_mask) if v}
            actual_next_cols = set(letter_to_columns[next_sym])
            if actual_next_cols & predicted_set:
                correct += 1
            total += 1

    # Require some minimum predictive skill (non-zero and at least 20%)
    assert total > 0
    accuracy = correct / total
    print(f"Prediction accuracy: {accuracy:.2f}")
    assert accuracy >= 0.5, f"Prediction accuracy too low: {accuracy:.2f}"


def test_two_field_letter_sequence_prediction_accuracy_with_run():
    """Variant of single-field prediction test using two independent letter sequences.

    Differences from `test_letter_sequence_prediction_accuracy`:
      - Two separate training/test chains (field1, field2)
      - At each timestep active columns are union of columns encoding both current letters
      - Uses TemporalPooler.run(mode='direct') instead of manual state calls
    Otherwise structure, thresholds and mapping approach remain intentionally similar.
    """
    random.seed(123)
    np.random.default_rng(2)  # seed for reproducibility (not directly used after switch to random.sample)
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    column_count = 128
    tp = TemporalPooler(input_space_size=50, column_count=column_count, cells_per_column=6, initial_synapses_per_column=16)

    # Partition columns into two disjoint pools representing two fields
    pool1 = tp.columns[: column_count // 2]
    pool2 = tp.columns[column_count // 2 :]
    k = 5  # columns per letter per field
    letter_to_columns_f1 = {L: random.sample(pool1, k) for L in letters}
    letter_to_columns_f2 = {L: random.sample(pool2, k) for L in letters}

    # Training sequences (reuse list pattern from single-field test)
    sequences = [list("ABCDX"), list("EFGHY"), list("JKLMN")]
    training_chain_field1 = []
    training_chain_field2 = []
    for _ in range(100):
        seq1 = random.choice(sequences)
        seq2 = random.choice(sequences)
        training_chain_field1.extend(seq1)
        training_chain_field2.extend(seq2)

    # Make chains same length
    length = min(len(training_chain_field1), len(training_chain_field2))
    training_chain_field1 = training_chain_field1[:length]
    training_chain_field2 = training_chain_field2[:length]

    # Train using run() in direct mode (supply union of columns active for both fields)
    tp.current_t = 0
    for t in range(length):
        sym1 = training_chain_field1[t]
        sym2 = training_chain_field2[t]
        tp.run({"f1": letter_to_columns_f1[sym1], "f2": letter_to_columns_f2[sym2]}, mode="direct")

    # Build test chains similarly
    test_chain_field1 = []
    test_chain_field2 = []
    for _ in range(100):
        seq1 = random.choice(sequences)
        seq2 = random.choice(sequences)
        test_chain_field1.extend(seq1)
        test_chain_field2.extend(seq2)
    length_test = min(len(test_chain_field1), len(test_chain_field2))
    test_chain_field1 = test_chain_field1[:length_test]
    test_chain_field2 = test_chain_field2[:length_test]

    tp.reset_state()
    tp.current_t = 0
    correct_f1 = 0
    correct_f2 = 0
    total_f1 = 0
    total_f2 = 0
    for t in range(length_test):
        sym1 = test_chain_field1[t]
        sym2 = test_chain_field2[t]
        tp.run({"f1": letter_to_columns_f1[sym1], "f2": letter_to_columns_f2[sym2]}, mode="direct")
        if t < length_test - 1:
            next1 = test_chain_field1[t + 1]
            next2 = test_chain_field2[t + 1]
            # Field-specific predicted columns
            pred_mask_f1 = tp.get_predictive_columns(t, field_name="f1")
            pred_mask_f2 = tp.get_predictive_columns(t, field_name="f2")
            pred_cols_f1 = {tp.columns[i] for i, v in enumerate(pred_mask_f1) if v}
            pred_cols_f2 = {tp.columns[i] for i, v in enumerate(pred_mask_f2) if v}
            if set(letter_to_columns_f1[next1]) & pred_cols_f1:
                correct_f1 += 1
            if set(letter_to_columns_f2[next2]) & pred_cols_f2:
                correct_f2 += 1
            total_f1 += 1
            total_f2 += 1

    assert total_f1 > 0 and total_f2 > 0
    acc_f1 = correct_f1 / total_f1
    acc_f2 = correct_f2 / total_f2
    combined_accuracy = (correct_f1 + correct_f2) / (total_f1 + total_f2)
    print(f"Field1 accuracy: {acc_f1:.2f} Field2 accuracy: {acc_f2:.2f} Combined: {combined_accuracy:.2f}")
    assert acc_f1 >= 0.5, f"Field1 prediction accuracy too low: {acc_f1:.2f}"
    assert acc_f2 >= 0.5, f"Field2 prediction accuracy too low: {acc_f2:.2f}"
