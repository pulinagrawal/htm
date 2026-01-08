"""
Unit tests for the SpatialPooler class.
Tests spatial pooler functionality including initialization, overlap computation,
inhibition, column activation, and learning mechanisms.
"""
import numpy as np
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from spatial_pooler import (
    SpatialPooler,
    ProximalSynapse,
    Column,
    CONNECTED_PERM,
    MIN_OVERLAP,
)


def test_spatial_pooler_initialization():
    """Test that SpatialPooler initializes correctly with expected structure."""
    num_columns = 64
    input_size = 100
    sp = SpatialPooler(num_columns=num_columns, input_size=input_size)
    
    assert sp.num_columns == num_columns
    assert sp.input_size == input_size
    assert len(sp.columns) == num_columns
    
    # Each column should have potential synapses
    for col in sp.columns:
        assert isinstance(col, Column)
        assert len(col.potential_synapses) > 0
        assert all(isinstance(syn, ProximalSynapse) for syn in col.potential_synapses)


def test_column_initialization():
    """Test that Column objects are properly initialized."""
    synapses = [
        ProximalSynapse(source_input=0, permanence=0.6),
        ProximalSynapse(source_input=1, permanence=0.4),
        ProximalSynapse(source_input=2, permanence=0.7),
    ]
    col = Column(potential_synapses=synapses, position=(0, 0))
    
    assert col.boost == 1.0
    assert col.active_duty_cycle == 0.0
    assert col.overlap == 0.0
    assert col.active == False
    # Connected synapses should only include those with permanence > CONNECTED_PERM
    assert len(col.connected_synapses) == 2  # permanences 0.6 and 0.7


def test_compute_overlap():
    """Test overlap computation with input vector."""
    synapses = [
        ProximalSynapse(source_input=0, permanence=0.6),
        ProximalSynapse(source_input=2, permanence=0.7),
        ProximalSynapse(source_input=5, permanence=0.55),
        ProximalSynapse(source_input=7, permanence=0.65),
    ]
    col = Column(potential_synapses=synapses)
    
    # Input activates positions 0, 2, 5 (3 connected synapses)
    input_vec = np.zeros(10, dtype=int)
    input_vec[[0, 2, 5]] = 1
    
    col.compute_overlap(input_vec)
    # Overlap should be 3 (synapses at positions 0, 2, and 5)
    # Since boost=1.0 and overlap >= MIN_OVERLAP(3), overlap = 3.0
    assert col.overlap == 3.0
    
    # Test with insufficient overlap (< MIN_OVERLAP)
    input_vec2 = np.zeros(10, dtype=int)
    input_vec2[0] = 1  # Only 1 connected synapse active
    col.compute_overlap(input_vec2)
    assert col.overlap == 0.0  # Below MIN_OVERLAP


def test_compute_overlap_with_boost():
    """Test that boost factor is applied to overlap computation."""
    synapses = [
        ProximalSynapse(source_input=i, permanence=0.6)
        for i in range(5)
    ]
    col = Column(potential_synapses=synapses)
    col.boost = 2.0
    
    # Activate all 5 inputs
    input_vec = np.ones(10, dtype=int)
    col.compute_overlap(input_vec)
    
    # Overlap = 5 * boost(2.0) = 10.0
    assert col.overlap == 10.0


def test_inhibition_basic():
    """Test basic inhibition mechanism."""
    sp = SpatialPooler(num_columns=16, input_size=32)
    
    # Set up overlaps manually for testing
    for i, col in enumerate(sp.columns):
        col.overlap = float(i)  # Overlaps: 0, 1, 2, ..., 15
    
    active_cols = sp.inhibition(sp.columns, inhibition_radius=1.0)
    
    # With sufficient overlap, some columns should be active
    assert len(active_cols) > 0
    # All active columns should have overlap > 0
    assert all(col.overlap > 0 for col in active_cols)


def test_set_active_columns():
    """Test setting active column state from binary vector."""
    sp = SpatialPooler(num_columns=10, input_size=20)
    
    # Set columns 0, 2, 5 as active
    column_state = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    sp.set_active_columns(column_state)
    
    assert sp.columns[0].active == True
    assert sp.columns[1].active == False
    assert sp.columns[2].active == True
    assert sp.columns[5].active == True


def test_run_method():
    """Test the run method which computes active columns."""
    sp = SpatialPooler(num_columns=16, input_size=32)
    
    # Create an input with some active bits
    input_vec = np.zeros(32, dtype=int)
    input_vec[[0, 5, 10, 15, 20]] = 1
    
    active_cols = sp.run(input_vec, learn=False)
    
    # Should return a list of Column objects
    assert isinstance(active_cols, list)
    assert all(isinstance(col, Column) for col in active_cols)
    
    # Active columns should have their .active flag set
    for col in sp.columns:
        if col in active_cols:
            assert col.active == True


def test_learning_increases_permanence_for_active_inputs():
    """Test that learning increases permanence for active input bits."""
    sp = SpatialPooler(num_columns=4, input_size=10)
    
    # Create a simple input
    input_vec = np.zeros(10, dtype=int)
    input_vec[[0, 1, 2]] = 1
    
    # Get initial permanences for first column
    col = sp.columns[0]
    initial_perms = {syn.source_input: syn.permanence for syn in col.potential_synapses}
    
    # Set column as active and run learning
    col.active = True
    sp.current_input_indices = {0, 1, 2}
    sp.learn(learning_rate=0.1)
    
    # Check permanences changed appropriately
    for syn in col.potential_synapses:
        if syn.source_input in [0, 1, 2]:
            # Active inputs should have increased permanence
            assert syn.permanence >= initial_perms[syn.source_input]
        else:
            # Inactive inputs should have decreased permanence
            assert syn.permanence <= initial_perms[syn.source_input]


def test_learning_updates_connected_synapses():
    """Test that learning updates the list of connected synapses."""
    sp = SpatialPooler(num_columns=4, input_size=10)
    col = sp.columns[0]
    
    # Use the first potential synapse and set it just below threshold
    assert len(col.potential_synapses) > 0, "Column should have potential synapses"
    target_synapse = col.potential_synapses[0]
    target_input_idx = target_synapse.source_input
    target_synapse.permanence = 0.48
    
    # Rebuild connected synapses list after manual modification
    col.connected_synapses = [s for s in col.potential_synapses if s.permanence > CONNECTED_PERM]
    
    # Check if target synapse is connected initially (should not be)
    synapse_connected_before = any(
        syn.source_input == target_input_idx for syn in col.connected_synapses
    )
    assert not synapse_connected_before, f"Synapse for input {target_input_idx} should not be connected initially"
    
    # Activate column and the target input bit, then learn
    col.active = True
    sp.current_input_indices = {target_input_idx}
    sp.learn(learning_rate=0.1)
    
    # The target synapse should now be connected (0.48 + 0.1 = 0.58 > 0.5)
    synapse_connected_after = any(
        syn.source_input == target_input_idx for syn in col.connected_synapses
    )
    assert synapse_connected_after, f"Synapse for input {target_input_idx} should be connected after learning"


def test_combine_input_fields_single_array():
    """Test combining input fields from a single array."""
    sp = SpatialPooler(num_columns=4, input_size=10)
    
    input_vec = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 1], dtype=int)
    combined = sp.combine_input_fields(input_vec)
    
    assert combined.shape[0] == 10
    assert np.array_equal(combined, input_vec)


def test_combine_input_fields_list():
    """Test combining input fields from a list of arrays."""
    sp = SpatialPooler(num_columns=4, input_size=10)
    
    field1 = np.array([1, 0, 1, 0], dtype=int)
    field2 = np.array([1, 1, 0, 0, 1, 1], dtype=int)
    combined = sp.combine_input_fields([field1, field2])
    
    assert combined.shape[0] == 10
    assert np.array_equal(combined, np.concatenate([field1, field2]))


def test_combine_input_fields_dict():
    """Test combining input fields from a dictionary."""
    sp = SpatialPooler(num_columns=4, input_size=10)
    
    input_dict = {
        'field1': np.array([1, 0, 1, 0], dtype=int),
        'field2': np.array([1, 1, 0, 0, 1, 1], dtype=int)
    }
    combined = sp.combine_input_fields(input_dict)
    
    assert combined.shape[0] == 10
    # Check field ranges are set
    assert 'field1' in sp.field_ranges
    assert 'field2' in sp.field_ranges
    assert sp.field_ranges['field1'] == (0, 4)
    assert sp.field_ranges['field2'] == (4, 10)


def test_integration_repeated_input_increases_column_activation():
    """
    Main integration test: When presented with a long sequence of inputs,
    and setting specific random columns active for certain inputs,
    those columns should be more likely to be active next time that input is presented.
    
    This tests the core learning mechanism of the spatial pooler.
    """
    np.random.seed(42)  # For reproducibility
    
    # Create spatial pooler
    num_columns = 64
    input_size = 100
    sp = SpatialPooler(num_columns=num_columns, input_size=input_size, 
                       potential_receptive_field_pct=0.5)
    
    # Create a set of distinct input patterns
    num_patterns = 5
    patterns = []
    for i in range(num_patterns):
        pattern = np.zeros(input_size, dtype=int)
        # Each pattern has 20 random active bits
        active_indices = np.random.choice(input_size, size=20, replace=False)
        pattern[active_indices] = 1
        patterns.append(pattern)
    
    # Define which columns we want to force active for each pattern
    forced_columns_per_pattern = {}
    for i in range(num_patterns):
        # For each pattern, select 5 random columns to force active
        forced_cols = np.random.choice(num_columns, size=5, replace=False)
        forced_columns_per_pattern[i] = set(forced_cols)
    
    # Training phase: Present patterns multiple times with forced activations
    num_training_iterations = 50
    for iteration in range(num_training_iterations):
        for pattern_idx, pattern in enumerate(patterns):
            # Run spatial pooler
            active_cols = sp.run(pattern, learn=False)
            
            # Force specific columns to be active
            forced_indices = forced_columns_per_pattern[pattern_idx]
            for idx in forced_indices:
                sp.columns[idx].active = True
            
            # Mark the active columns
            for col_idx, col in enumerate(sp.columns):
                if col_idx in forced_indices or col in active_cols:
                    col.active = True
                else:
                    col.active = False
            
            # Run learning
            sp.current_input_indices = set(np.where(pattern > 0)[0])
            sp.learn(learning_rate=0.05)
    
    # Testing phase: Re-present each pattern and check if forced columns activate
    activation_scores = {i: [] for i in range(num_patterns)}
    
    num_test_iterations = 20
    for iteration in range(num_test_iterations):
        for pattern_idx, pattern in enumerate(patterns):
            # Run spatial pooler without forcing
            active_cols = sp.run(pattern, learn=False)
            active_col_indices = {sp.columns.index(col) for col in active_cols}
            
            # Check how many of the forced columns are now naturally active
            forced_indices = forced_columns_per_pattern[pattern_idx]
            num_forced_active = len(forced_indices.intersection(active_col_indices))
            activation_rate = num_forced_active / len(forced_indices)
            activation_scores[pattern_idx].append(activation_rate)
    
    # Verify that for each pattern, the forced columns have higher activation rate
    # than random chance
    for pattern_idx in range(num_patterns):
        avg_activation_rate = np.mean(activation_scores[pattern_idx])
        
        # Random chance would be approximately (num_active_cols / num_columns)
        # We expect significantly higher than random chance
        # Let's say at least 30% of the forced columns should be active
        print(f"Pattern {pattern_idx}: Average activation rate = {avg_activation_rate:.2%}")
        assert avg_activation_rate > 0.2, \
            f"Pattern {pattern_idx} failed: activation rate {avg_activation_rate:.2%} too low"
    
    # Overall, the average across all patterns should be quite good
    overall_avg = np.mean([np.mean(scores) for scores in activation_scores.values()])
    print(f"Overall average activation rate: {overall_avg:.2%}")
    assert overall_avg > 0.25, f"Overall activation rate {overall_avg:.2%} is too low"


def test_kth_score():
    """Test the kth_score method for finding k-th highest overlap."""
    sp = SpatialPooler(num_columns=10, input_size=20)
    
    # Create neighbors with specific overlaps
    neighbors = []
    for i in range(5):
        col = Column(potential_synapses=[], position=(i, 0))
        col.overlap = float(i * 2)  # Overlaps: 0, 2, 4, 6, 8
        neighbors.append(col)
    
    # Test with sparsity = 0.4 (40%) -> k = ceil(0.4 * 5) = 2
    k_score = sp.kth_score(neighbors, sparsity=0.4)
    # 2nd highest should be 6.0 (sorted descending: 8, 6, 4, 2, 0)
    assert k_score == 6.0
    
    # Test with empty neighbors
    k_score_empty = sp.kth_score([], sparsity=0.5)
    assert k_score_empty == 0
    
    # Test with sparsity requiring more than available
    k_score_high = sp.kth_score(neighbors, sparsity=2.0)
    assert k_score_high == 0.0  # Should return lowest overlap


if __name__ == "__main__":
    # Run tests
    test_spatial_pooler_initialization()
    test_column_initialization()
    test_compute_overlap()
    test_compute_overlap_with_boost()
    test_inhibition_basic()
    test_set_active_columns()
    test_run_method()
    test_learning_increases_permanence_for_active_inputs()
    test_learning_updates_connected_synapses()
    test_combine_input_fields_single_array()
    test_combine_input_fields_list()
    test_combine_input_fields_dict()
    test_kth_score()
    test_integration_repeated_input_increases_column_activation()
    print("All tests passed!")
