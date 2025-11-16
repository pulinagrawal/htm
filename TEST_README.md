# HTM Building Blocks - Test Suite

This directory contains comprehensive unit tests for the HTM (Hierarchical Temporal Memory) building blocks and layer architecture.

## Running the Tests

To run all tests:

```bash
python3 test_building_blocks.py
```

Or using Python's unittest module:

```bash
python3 -m unittest test_building_blocks.py -v
```

## Test Coverage

The test suite includes 48 comprehensive tests covering:

### 1. Basic Building Blocks (7 tests)
- Cell creation and state management
- DistalSynapse and ProximalSynapse functionality
- Segment active/matching synapse detection
- Column overlap computation and cell management

### 2. SpatialPoolerLayer (6 tests)
- Initialization and configuration
- Cell activation and sparsity
- Learning and permanence updates
- Reset functionality
- Dynamic column addition
- Long sequence stability (50 steps)

### 3. TemporalMemoryLayer (6 tests)
- Column bursting behavior
- Predictive cell activation
- Segment creation during learning
- State reset
- Dynamic growth
- Sequence learning (50+ steps)

### 4. CustomDistalLayer (5 tests)
- Fire-together-wire-together learning
- Layer connectivity
- State management
- Dynamic cell addition
- Long sequence processing (30 steps)

### 5. SparseyInspiredSpatialPooler (5 tests)
- Neighborhood organization
- Activation constraints per neighborhood
- Reset functionality
- Dynamic neighborhood addition
- Long sequence processing (40 steps)

### 6. Layer Connectivity (3 tests)
- Multi-layer connection mechanism
- State propagation between layers
- Multi-layer network sequences

### 7. Dynamic Growth (4 tests)
- Runtime addition of columns/cells
- Growth across all layer types

### 8. Long Sequences (4 tests)
- 100-step spatial pooler stability
- 100-step temporal memory patterns
- 80-step multi-layer networks
- Pattern learning over many iterations

### 9. Edge Cases (5 tests)
- Empty and full input handling
- Extreme activation scenarios
- Learning rate effects

## Test Results

All tests pass successfully:
- ✅ 48 tests
- ✅ 0 failures
- ✅ 0 errors

## Requirements

- Python 3.7+
- NumPy

Install dependencies:
```bash
pip install numpy
```

## Test Organization

Tests are organized by component:
- `TestBasicBuildingBlocks` - Core data structures
- `TestSpatialPoolerLayer` - Spatial pooling functionality
- `TestTemporalMemoryLayer` - Temporal memory functionality
- `TestCustomDistalLayer` - Custom distal learning
- `TestSparseyInspiredSpatialPooler` - Sparsey-inspired pooling
- `TestLayerConnectivity` - Multi-layer integration
- `TestDynamicGrowth` - Runtime growth capabilities
- `TestLongerSequences` - Long sequence robustness
- `TestEdgeCases` - Boundary conditions

## Adding New Tests

To add new tests:

1. Create a new test class inheriting from `unittest.TestCase`
2. Add test methods starting with `test_`
3. Add the test class to the suite in `run_tests()`

Example:
```python
class TestMyNewFeature(unittest.TestCase):
    def test_feature_behavior(self):
        # Test code here
        self.assertEqual(expected, actual)
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```bash
# Exit with non-zero code if tests fail
python3 test_building_blocks.py || exit 1
```

## Security

All tests have been validated with CodeQL security scanning:
- ✅ 0 security alerts
- ✅ No vulnerabilities detected

## Support

For issues or questions about the tests, please refer to the main README.md or open an issue on the repository.
