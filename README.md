# Flexible Neocortex Layer Architecture

A flexible, extensible layer-based architecture for neocortex modeling and Hierarchical Temporal Memory (HTM) research.

## Overview

This codebase provides a clean, decoupled architecture for creating and connecting layers of cells that model neocortical computation. It supports various layer types inspired by HTM theory and enables easy experimentation with novel learning mechanisms.

## Features

- **Flexible Layer Architecture**: Abstract base class allows easy creation of new layer types
- **Layer Connectivity**: Connect layers together and pass state between them
- **Dynamic Growth**: All layers support adding new cells/columns at runtime
- **Configurable Learning Rates**: Scale permanence changes for different learning dynamics
- **Multiple Layer Types**: Pre-built implementations for common patterns

## Layer Types

### 1. SpatialPoolerLayer

HTM-style spatial pooler that learns sparse distributed representations.

**Features:**
- Proximal synapse learning
- Global inhibition
- Overlap computation with boosting
- Configurable sparsity

**Example:**
```python
from building_blocks import SpatialPoolerLayer
import numpy as np

sp = SpatialPoolerLayer(
    input_size=100,
    num_columns=256,
    sparsity=0.02,
    learning_rate=1.0
)

input_vector = np.random.randint(0, 2, size=100).astype(float)
sp.set_input(input_vector)
sp.compute(learn=True)

print(f"Active cells: {len(sp.active_cells)}")
```

### 2. TemporalMemoryLayer

HTM-style temporal memory that learns sequences through distal dendrites.

**Features:**
- Predictive cells and bursting
- Distal dendrite learning
- Segment creation and adaptation
- Temporal sequence learning

**Example:**
```python
from building_blocks import TemporalMemoryLayer

tm = TemporalMemoryLayer(
    num_columns=256,
    cells_per_column=32,
    learning_rate=1.0
)

# Set which columns are active (typically from spatial pooler)
active_columns = {1, 5, 10, 23, 45}
tm.set_active_columns(active_columns)
tm.compute(learn=True)

print(f"Active cells: {len(tm.active_cells)}")
print(f"Predictive cells: {len(tm.predictive_cells)}")
```

### 3. CustomDistalLayer

A novel layer type with fire-together-wire-together learning.

**Features:**
- Hebbian-style plasticity
- Automatic segment creation
- Synapses connect to cells in other layers
- Simple, biologically-inspired learning

**Learning Rule:**
- If a cell is active AND has an active distal segment: strengthen that segment
- If a cell is active but has NO active segment: create new segment with synapses to currently active input cells

**Example:**
```python
from building_blocks import CustomDistalLayer, SpatialPoolerLayer

# Create input layer
input_layer = SpatialPoolerLayer(input_size=50, num_columns=128)

# Create custom distal layer
custom = CustomDistalLayer(
    num_cells=64,
    learning_rate=1.0
)

# Connect layers
custom.connect_input(input_layer)

# Compute (learning happens automatically)
custom.compute(learn=True)
```

### 4. SparseyInspiredSpatialPooler

Spatial pooler with neighborhood constraints inspired by Sparsey.

**Features:**
- Columns organized into neighborhoods
- Fixed percentage of columns active per neighborhood
- Independent inhibition within each neighborhood
- Localized competition

**Example:**
```python
from building_blocks import SparseyInspiredSpatialPooler

sparsey = SparseyInspiredSpatialPooler(
    input_size=100,
    num_neighborhoods=8,
    columns_per_neighborhood=16,
    active_pct_per_neighborhood=0.1  # 10% active
)

input_vector = np.random.randint(0, 2, size=100).astype(float)
sparsey.set_input(input_vector)
sparsey.compute(learn=True)
```

## Layer Connectivity

Layers can be connected to form hierarchies:

```python
# Create layers
sp_layer = SpatialPoolerLayer(input_size=100, num_columns=256)
tm_layer = TemporalMemoryLayer(num_columns=256, cells_per_column=32)
custom_layer = CustomDistalLayer(num_cells=64)

# Connect them
tm_layer.connect_input(sp_layer)
custom_layer.connect_input(tm_layer)

# State flows through the network
sp_layer.compute(learn=True)
# tm_layer can access sp_layer.get_active_cells()
tm_layer.compute(learn=True)
# custom_layer can access tm_layer.get_active_cells()
custom_layer.compute(learn=True)
```

## Dynamic Growth

All layers support adding new cells or columns:

```python
# Spatial Pooler: add columns
sp = SpatialPoolerLayer(input_size=100, num_columns=32)
new_column = sp.add_column()
print(f"Now has {sp.num_columns} columns")

# Temporal Memory: add columns (with cells)
tm = TemporalMemoryLayer(num_columns=64, cells_per_column=4)
new_column = tm.add_column()  # Adds column with 4 cells
print(f"Total cells: {len(tm.get_cells())}")

# Custom Distal: add individual cells
custom = CustomDistalLayer(num_cells=16)
new_cell = custom.add_cell()
print(f"Now has {custom.num_cells} cells")
```

## Learning Rate

All layers support a learning rate parameter that scales permanence changes:

```python
# Fast learning
fast_layer = SpatialPoolerLayer(
    input_size=100,
    num_columns=256,
    learning_rate=2.0  # 2x permanence changes
)

# Slow learning
slow_layer = TemporalMemoryLayer(
    num_columns=256,
    cells_per_column=32,
    learning_rate=0.5  # 0.5x permanence changes
)
```

## Setting Cell States

You can manually set which cells are active in a layer:

```python
layer = CustomDistalLayer(num_cells=100)

# Get some cells
cells_to_activate = set(layer.get_cells()[:10])

# Set them active
layer.set_active_cells(cells_to_activate)

# Check active state
print(f"Active cells: {len(layer.active_cells)}")
for cell in cells_to_activate:
    assert cell.active == True
```

## Building Blocks

The architecture is built on these core components:

- **Cell**: Individual computational unit with segments
- **Segment**: Distal dendrite segment with synapses
- **DistalSynapse**: Synapse connecting to another cell
- **ProximalSynapse**: Synapse connecting to input bit
- **Column**: Group of cells with proximal synapses (for spatial pooling)
- **Layer**: Abstract base class for all layer types

## Creating Custom Layers

To create a new layer type, inherit from `Layer` and implement the abstract methods:

```python
from building_blocks import Layer, Cell

class MyCustomLayer(Layer):
    def __init__(self, num_cells, name="MyLayer", learning_rate=1.0):
        super().__init__(name, learning_rate)
        self.cells_list = [Cell() for _ in range(num_cells)]
    
    def compute(self, learn=True):
        """Implement your computation logic"""
        # Your code here
        pass
    
    def reset(self):
        """Reset layer state"""
        self.active_cells.clear()
        for cell in self.cells_list:
            cell.active = False
    
    def get_cells(self):
        """Return all cells"""
        return self.cells_list.copy()
```

## Examples

See `example_usage.py` for comprehensive examples including:
- Basic HTM hierarchy (SP + TM)
- Custom distal layer usage
- Sparsey-inspired pooler
- Dynamic growth
- Multi-layer networks

Run the examples:
```bash
python3 example_usage.py
```

## Design Principles

This architecture follows these principles for research code:

1. **Simplicity**: Straightforward implementations that are easy to understand
2. **Flexibility**: Easy to extend with new layer types and learning rules
3. **Decoupling**: Layers are independent and connect via clean interfaces
4. **Readability**: Clear naming and documentation
5. **Extensibility**: Abstract base classes enable customization

## Constants

Key constants are defined in `building_blocks.py`:

- `CONNECTED_PERM = 0.5`: Permanence threshold for connected synapses
- `MIN_OVERLAP = 3`: Minimum overlap for column activation
- `ACTIVATION_THRESHOLD = 3`: Synapses needed for segment activation
- `LEARNING_THRESHOLD = 5`: Synapses needed for learning
- `INITIAL_PERMANENCE = 0.21`: Initial permanence for new synapses
- `PERMANENCE_INC = 0.01`: Permanence increase amount
- `PERMANENCE_DEC = 0.01`: Permanence decrease amount

These can be customized per layer via constructor parameters.

## Requirements

- Python 3.7+
- NumPy

## Installation

```bash
pip install numpy
```

## License

See repository license.

## Contributing

This is a research codebase designed for experimentation. Feel free to:
- Create new layer types
- Modify learning rules
- Experiment with different architectures
- Add new features

The modular design makes it easy to extend without breaking existing functionality.
