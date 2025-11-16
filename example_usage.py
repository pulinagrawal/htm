"""
Example usage of the flexible neocortex layer architecture.

This demonstrates:
1. Creating different types of layers
2. Connecting layers together
3. Setting layer states
4. Running computation with learning
5. Dynamic growth of cells/columns
"""

import numpy as np
from building_blocks import (
    SpatialPoolerLayer,
    TemporalMemoryLayer,
    CustomDistalLayer,
    SparseyInspiredSpatialPooler,
)


def example_basic_hierarchy():
    """Example: Basic HTM hierarchy with Spatial Pooler and Temporal Memory."""
    print("=" * 60)
    print("Example 1: Basic HTM Hierarchy")
    print("=" * 60)
    
    # Create layers
    input_size = 100
    sp_layer = SpatialPoolerLayer(
        input_size=input_size,
        num_columns=256,
        sparsity=0.02,
        name="SpatialPooler",
        learning_rate=1.0
    )
    
    tm_layer = TemporalMemoryLayer(
        num_columns=256,
        cells_per_column=32,
        name="TemporalMemory",
        learning_rate=1.0
    )
    
    # Connect layers: TM receives input from SP
    tm_layer.connect_input(sp_layer)
    
    print(f"Created {sp_layer.name} with {sp_layer.num_columns} columns")
    print(f"Created {tm_layer.name} with {tm_layer.num_columns} columns, "
          f"{tm_layer.cells_per_column} cells per column")
    print()
    
    # Run a sequence
    print("Running sequence...")
    for step in range(5):
        # Generate random input
        input_vector = np.random.randint(0, 2, size=input_size).astype(float)
        
        # Spatial pooling
        sp_layer.set_input(input_vector)
        sp_layer.compute(learn=True)
        
        # Get active columns from spatial pooler
        active_column_indices = {i for i, col in enumerate(sp_layer.columns) if col.active}
        
        # Temporal memory
        tm_layer.set_active_columns(active_column_indices)
        tm_layer.compute(learn=True)
        
        print(f"  Step {step}: {len(sp_layer.active_cells)} SP cells active, "
              f"{len(tm_layer.active_cells)} TM cells active, "
              f"{len(tm_layer.predictive_cells)} TM cells predictive")
    
    print("\n")


def example_custom_distal_layer():
    """Example: Custom distal layer with fire-together-wire-together learning."""
    print("=" * 60)
    print("Example 2: Custom Distal Layer")
    print("=" * 60)
    
    # Create input layer (spatial pooler)
    sp_layer = SpatialPoolerLayer(
        input_size=50,
        num_columns=128,
        sparsity=0.05,
        name="InputSP"
    )
    
    # Create custom distal layer
    custom_layer = CustomDistalLayer(
        num_cells=64,
        name="CustomDistal",
        learning_rate=1.0
    )
    
    # Connect layers
    custom_layer.connect_input(sp_layer)
    
    print(f"Created {sp_layer.name} with {sp_layer.num_columns} columns")
    print(f"Created {custom_layer.name} with {custom_layer.num_cells} cells")
    print()
    
    # Run computation
    print("Running computation with fire-together-wire-together learning...")
    for step in range(3):
        # Generate input
        input_vector = np.random.randint(0, 2, size=50).astype(float)
        
        # Spatial pooling
        sp_layer.set_input(input_vector)
        sp_layer.compute(learn=True)
        
        # Custom layer computation
        custom_layer.compute(learn=True)
        
        print(f"  Step {step}: {len(custom_layer.active_cells)} custom cells active")
        
        # Manually activate some cells to trigger learning
        if step == 0:
            cells_to_activate = set(custom_layer.cells_list[:5])
            custom_layer.set_active_cells(cells_to_activate)
            print(f"    (Manually set {len(cells_to_activate)} cells active for learning)")
    
    print("\n")


def example_sparsey_pooler():
    """Example: Sparsey-inspired spatial pooler with neighborhood constraints."""
    print("=" * 60)
    print("Example 3: Sparsey-Inspired Spatial Pooler")
    print("=" * 60)
    
    # Create Sparsey pooler
    sparsey = SparseyInspiredSpatialPooler(
        input_size=100,
        num_neighborhoods=8,
        columns_per_neighborhood=16,
        active_pct_per_neighborhood=0.1,  # 10% active per neighborhood
        name="SparseyPooler"
    )
    
    print(f"Created {sparsey.name} with {sparsey.num_neighborhoods} neighborhoods")
    print(f"Each neighborhood has {sparsey.columns_per_neighborhood} columns")
    print(f"Active percentage per neighborhood: {sparsey.active_pct_per_neighborhood * 100}%")
    print()
    
    # Run computation
    print("Running computation...")
    for step in range(3):
        input_vector = np.random.randint(0, 2, size=100).astype(float)
        sparsey.set_input(input_vector)
        sparsey.compute(learn=True)
        
        # Count active columns per neighborhood
        active_per_neighborhood = []
        for neighborhood in sparsey.neighborhoods:
            active_count = sum(1 for col in neighborhood if col.active)
            active_per_neighborhood.append(active_count)
        
        print(f"  Step {step}: Total {len(sparsey.active_cells)} active cells")
        print(f"    Active per neighborhood: {active_per_neighborhood}")
    
    print("\n")


def example_dynamic_growth():
    """Example: Dynamic growth of cells and columns."""
    print("=" * 60)
    print("Example 4: Dynamic Growth")
    print("=" * 60)
    
    # Create small spatial pooler
    sp_layer = SpatialPoolerLayer(
        input_size=50,
        num_columns=32,
        name="GrowingSP"
    )
    
    print(f"Initial {sp_layer.name}: {sp_layer.num_columns} columns")
    
    # Add new columns
    for i in range(3):
        new_col = sp_layer.add_column()
        print(f"  Added column {sp_layer.num_columns - 1} at position {new_col.position}")
    
    print(f"After growth: {sp_layer.num_columns} columns")
    print()
    
    # Create temporal memory layer
    tm_layer = TemporalMemoryLayer(
        num_columns=64,
        cells_per_column=4,
        name="GrowingTM"
    )
    
    print(f"Initial {tm_layer.name}: {tm_layer.num_columns} columns, "
          f"{tm_layer.cells_per_column} cells per column")
    print(f"  Total cells: {len(tm_layer.get_cells())}")
    
    # Add new columns
    for i in range(2):
        new_col = tm_layer.add_column()
        print(f"  Added column with {len(new_col.cells)} cells")
    
    print(f"After growth: {tm_layer.num_columns} columns")
    print(f"  Total cells: {len(tm_layer.get_cells())}")
    print()
    
    # Custom distal layer
    custom_layer = CustomDistalLayer(num_cells=16, name="GrowingCustom")
    print(f"Initial {custom_layer.name}: {custom_layer.num_cells} cells")
    
    for i in range(5):
        custom_layer.add_cell()
    
    print(f"After growth: {custom_layer.num_cells} cells")
    print()


def example_multi_layer_network():
    """Example: Multi-layer network with various layer types."""
    print("=" * 60)
    print("Example 5: Multi-Layer Network")
    print("=" * 60)
    
    # Layer 1: Sparsey pooler as input processor
    layer1 = SparseyInspiredSpatialPooler(
        input_size=100,
        num_neighborhoods=4,
        columns_per_neighborhood=8,
        active_pct_per_neighborhood=0.15,
        name="Layer1_Sparsey",
        learning_rate=0.8
    )
    
    # Layer 2: Temporal memory
    layer2 = TemporalMemoryLayer(
        num_columns=32,
        cells_per_column=16,
        name="Layer2_TM",
        learning_rate=1.0
    )
    
    # Layer 3: Custom distal layer
    layer3 = CustomDistalLayer(
        num_cells=24,
        name="Layer3_Custom",
        learning_rate=1.2
    )
    
    # Connect layers
    layer2.connect_input(layer1)
    layer3.connect_input(layer2)
    
    print(f"Created 3-layer network:")
    print(f"  {layer1.name}: {layer1.num_neighborhoods} neighborhoods")
    print(f"  {layer2.name}: {layer2.num_columns} columns × {layer2.cells_per_column} cells")
    print(f"  {layer3.name}: {layer3.num_cells} cells")
    print()
    
    # Run a sequence
    print("Running multi-layer computation...")
    for step in range(3):
        # Generate input
        input_vector = np.random.randint(0, 2, size=100).astype(float)
        
        # Layer 1: Sparsey pooler
        layer1.set_input(input_vector)
        layer1.compute(learn=True)
        
        # Layer 2: Temporal memory
        active_cols = {i for i, neighborhood in enumerate(layer1.neighborhoods)
                      for j, col in enumerate(neighborhood) if col.active}
        # Map to column indices
        active_col_indices = set()
        for i, neighborhood in enumerate(layer1.neighborhoods):
            for j, col in enumerate(neighborhood):
                if col.active:
                    col_idx = i * layer1.columns_per_neighborhood + j
                    if col_idx < layer2.num_columns:
                        active_col_indices.add(col_idx)
        
        layer2.set_active_columns(active_col_indices)
        layer2.compute(learn=True)
        
        # Layer 3: Custom distal
        layer3.compute(learn=True)
        
        print(f"  Step {step}:")
        print(f"    Layer1: {len(layer1.active_cells)} active")
        print(f"    Layer2: {len(layer2.active_cells)} active, {len(layer2.predictive_cells)} predictive")
        print(f"    Layer3: {len(layer3.active_cells)} active")
    
    print("\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FLEXIBLE NEOCORTEX LAYER ARCHITECTURE - EXAMPLES")
    print("=" * 60 + "\n")
    
    # Run all examples
    example_basic_hierarchy()
    example_custom_distal_layer()
    example_sparsey_pooler()
    example_dynamic_growth()
    example_multi_layer_network()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
