import numpy as np
from abc import ABC, abstractmethod
from typing import (
    List,
    Set,
    Dict,
    Tuple,
    Optional,
    Sequence,
    Union,
    Any,
)

# Constants
CONNECTED_PERM = 0.5  # Permanence threshold for a synapse to be considered connected
MIN_OVERLAP = 3       # Minimum overlap to be considered during inhibition
DESIRED_LOCAL_SPARSITY = 0.02  # Desired local sparsity for inhibition
INHIBITION_RADIUS = 1.0  # Sets the locality radius for desired sparsity
ACTIVATION_THRESHOLD = 3  # Number of active connected synapses for a segment to be active
LEARNING_THRESHOLD = 5  # Threshold of active synapses for a segment to be considered for learning
INITIAL_PERMANENCE = 0.21  # Initial permanence for new synapses
PERMANENCE_INC = 0.01  # Amount by which synapses are incremented during learning
PERMANENCE_DEC = 0.01  # Amount by which synapses are decremented during learning


# ===== Basic Building Blocks =====

class Cell:
    """Single cell within a column or layer.
    
    Holds a (possibly empty) list of distal segments used for temporal learning.
    """
    
    def __init__(self) -> None:
        self.segments: List['Segment'] = []
        self.active: bool = False
        self.predictive: bool = False
        self.learning: bool = False
        
    def __repr__(self) -> str:
        return f"Cell(id={id(self)})"


class DistalSynapse:
    """Distal synapse connecting to a source cell."""
    
    def __init__(self, source_cell: Cell, permanence: float) -> None:
        self.source_cell: Cell = source_cell
        self.permanence: float = permanence


class Segment:
    """Distal segment composed of synapses to cells."""
    
    def __init__(self, synapses: Optional[List[DistalSynapse]] = None) -> None:
        self.synapses: List[DistalSynapse] = synapses if synapses is not None else []
        self.sequence_segment: bool = False  # True if learned in a predictive context
        
    def active_synapses(self, active_cells: Set[Cell], connected_perm: float = CONNECTED_PERM) -> List[DistalSynapse]:
        """Return connected synapses whose source cell is active."""
        return [syn for syn in self.synapses 
                if syn.source_cell in active_cells and syn.permanence >= connected_perm]
    
    def matching_synapses(self, active_cells: Set[Cell]) -> List[DistalSynapse]:
        """Return synapses whose source cell is active (ignores permanence threshold)."""
        return [syn for syn in self.synapses if syn.source_cell in active_cells]


class ProximalSynapse:
    """Proximal synapse connecting to an input bit."""
    
    def __init__(self, source_input: int, permanence: float) -> None:
        self.source_input: int = source_input
        self.permanence: float = permanence


class Column:
    """Column containing cells and proximal synapses for spatial pooling."""
    
    def __init__(self, 
                 potential_synapses: List[ProximalSynapse], 
                 position: Tuple[int, int] = (0, 0),
                 cells_per_column: int = 1) -> None:
        self.position: Tuple[int, int] = position
        self.potential_synapses: List[ProximalSynapse] = potential_synapses
        self.boost: float = 1.0
        self.active_duty_cycle: float = 0.0
        self.overlap_duty_cycle: float = 0.0
        self.min_duty_cycle: float = 0.01
        self.connected_synapses: List[ProximalSynapse] = []
        self.overlap: float = 0.0
        self.active: bool = False
        self.cells: List[Cell] = [Cell() for _ in range(cells_per_column)]
        self._update_connected_synapses()
        
    def _update_connected_synapses(self, connected_perm: float = CONNECTED_PERM) -> None:
        """Update the list of connected synapses based on permanence threshold."""
        self.connected_synapses = [s for s in self.potential_synapses 
                                   if s.permanence >= connected_perm]
    
    def compute_overlap(self, input_vector: np.ndarray, min_overlap: int = MIN_OVERLAP) -> None:
        """Compute overlap with current binary input vector and apply boost."""
        overlap = sum(1 for s in self.connected_synapses 
                     if s.source_input < len(input_vector) and input_vector[s.source_input])
        self.overlap = float(overlap * self.boost) if overlap >= min_overlap else 0.0
    
    def add_cell(self) -> Cell:
        """Add a new cell to this column."""
        new_cell = Cell()
        self.cells.append(new_cell)
        return new_cell


# ===== Layer Base Class =====

class Layer(ABC):
    """Abstract base class for all layer types.
    
    A layer is a collection of cells that can receive input from other layers
    and produce output that can be sent to other layers.
    """
    
    def __init__(self, name: str = "Layer", learning_rate: float = 1.0) -> None:
        self.name: str = name
        self.learning_rate: float = learning_rate  # Scales permanence changes
        self.input_layers: List['Layer'] = []
        self.active_cells: Set[Cell] = set()
        self.winner_cells: Set[Cell] = set()
        
    @abstractmethod
    def compute(self, learn: bool = True) -> None:
        """Compute layer activations."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset layer state."""
        pass
    
    @abstractmethod
    def get_cells(self) -> List[Cell]:
        """Return all cells in the layer."""
        pass
    
    def connect_input(self, layer: 'Layer') -> None:
        """Connect another layer as input to this layer."""
        if layer not in self.input_layers:
            self.input_layers.append(layer)
    
    def get_active_cells(self) -> Set[Cell]:
        """Return currently active cells."""
        return self.active_cells.copy()
    
    def set_active_cells(self, cells: Set[Cell]) -> None:
        """Set which cells are active."""
        # Clear previous state
        for cell in self.get_cells():
            cell.active = False
        # Set new active cells
        self.active_cells = cells.copy()
        for cell in self.active_cells:
            cell.active = True
    
    def _adjust_permanence(self, synapse: Union[DistalSynapse, ProximalSynapse], 
                          increase: bool) -> None:
        """Adjust synapse permanence by learning rate."""
        if increase:
            synapse.permanence = min(1.0, synapse.permanence + PERMANENCE_INC * self.learning_rate)
        else:
            synapse.permanence = max(0.0, synapse.permanence - PERMANENCE_DEC * self.learning_rate)


# ===== Temporal Memory Layer =====

class TemporalMemoryLayer(Layer):
    """Temporal Memory layer implementation following HTM principles.
    
    Learns temporal sequences through distal dendrites on cells.
    """
    
    def __init__(self, 
                 num_columns: int,
                 cells_per_column: int = 32,
                 name: str = "TemporalMemory",
                 learning_rate: float = 1.0,
                 activation_threshold: int = ACTIVATION_THRESHOLD,
                 learning_threshold: int = LEARNING_THRESHOLD,
                 max_new_synapse_count: int = 20,
                 initial_permanence: float = INITIAL_PERMANENCE,
                 sample_size: int = 20) -> None:
        super().__init__(name, learning_rate)
        self.num_columns = num_columns
        self.cells_per_column = cells_per_column
        self.activation_threshold = activation_threshold
        self.learning_threshold = learning_threshold
        self.max_new_synapse_count = max_new_synapse_count
        self.initial_permanence = initial_permanence
        self.sample_size = sample_size
        
        # Create columns with cells
        self.columns: List[Column] = []
        for i in range(num_columns):
            col = Column([], position=(i, 0), cells_per_column=cells_per_column)
            self.columns.append(col)
        
        self.active_columns: Set[int] = set()
        self.predictive_cells: Set[Cell] = set()
        self.prev_active_cells: Set[Cell] = set()
        self.prev_winner_cells: Set[Cell] = set()
        
    def compute(self, learn: bool = True) -> None:
        """Compute temporal memory activations."""
        # Phase 1: Activate cells
        self._activate_cells(learn)
        
        # Phase 2: Activate dendrites for next timestep
        self._activate_dendrites()
        
    def set_active_columns(self, active_column_indices: Set[int]) -> None:
        """Set which columns are active (typically from spatial pooler)."""
        self.active_columns = active_column_indices.copy()
        
    def _activate_cells(self, learn: bool) -> None:
        """Activate cells based on active columns and predictions."""
        new_active_cells: Set[Cell] = set()
        new_winner_cells: Set[Cell] = set()
        
        for col_idx in self.active_columns:
            if col_idx >= len(self.columns):
                continue
            column = self.columns[col_idx]
            
            # Check for predicted cells in this column
            predicted_cells = [cell for cell in column.cells if cell in self.predictive_cells]
            
            if predicted_cells:
                # Activate predicted cells
                for cell in predicted_cells:
                    new_active_cells.add(cell)
                    new_winner_cells.add(cell)
                    
                    if learn:
                        # Learn on active segments
                        for segment in cell.segments:
                            active_syns = segment.active_synapses(self.prev_active_cells)
                            if len(active_syns) >= self.activation_threshold:
                                self._adapt_segment(segment, self.prev_active_cells, True)
            else:
                # Burst column - activate all cells
                for cell in column.cells:
                    new_active_cells.add(cell)
                
                # Choose a winner cell
                winner_cell = self._get_best_matching_cell(column, self.prev_active_cells)
                new_winner_cells.add(winner_cell)
                
                if learn:
                    # Learn on winner cell
                    self._learn_on_cell(winner_cell, self.prev_active_cells)
        
        # Update state
        self.prev_active_cells = self.active_cells.copy()
        self.prev_winner_cells = self.winner_cells.copy()
        self.set_active_cells(new_active_cells)
        self.winner_cells = new_winner_cells
        
    def _activate_dendrites(self) -> None:
        """Update predictive cells for next timestep."""
        new_predictive_cells: Set[Cell] = set()
        
        for column in self.columns:
            for cell in column.cells:
                for segment in cell.segments:
                    active_syns = segment.active_synapses(self.active_cells)
                    if len(active_syns) >= self.activation_threshold:
                        new_predictive_cells.add(cell)
                        break
        
        self.predictive_cells = new_predictive_cells
        
    def _get_best_matching_cell(self, column: Column, active_cells: Set[Cell]) -> Cell:
        """Find cell with best matching segment, or least used cell."""
        best_cell = None
        best_segment = None
        max_matching = -1
        
        for cell in column.cells:
            for segment in cell.segments:
                matching_syns = segment.matching_synapses(active_cells)
                if len(matching_syns) > max_matching:
                    max_matching = len(matching_syns)
                    best_cell = cell
                    best_segment = segment
        
        if best_cell is None:
            # Use least used cell
            best_cell = min(column.cells, key=lambda c: len(c.segments))
        
        return best_cell
    
    def _learn_on_cell(self, cell: Cell, active_cells: Set[Cell]) -> None:
        """Create or adapt segment on cell to learn pattern."""
        # Find best matching segment
        best_segment = None
        max_matching = -1
        
        for segment in cell.segments:
            matching_syns = segment.matching_synapses(active_cells)
            if len(matching_syns) > max_matching:
                max_matching = len(matching_syns)
                best_segment = segment
        
        if best_segment is None or max_matching < self.learning_threshold:
            # Create new segment
            best_segment = Segment()
            cell.segments.append(best_segment)
        
        # Adapt segment
        self._adapt_segment(best_segment, active_cells, False)
        
    def _adapt_segment(self, segment: Segment, active_cells: Set[Cell], 
                       reinforce_only: bool = False) -> None:
        """Adapt segment synapses based on active cells."""
        # Get synapse sources
        synapse_sources = {syn.source_cell for syn in segment.synapses}
        
        # Strengthen synapses to active cells
        for syn in segment.synapses:
            if syn.source_cell in active_cells:
                self._adjust_permanence(syn, increase=True)
            elif not reinforce_only:
                self._adjust_permanence(syn, increase=False)
        
        # Add new synapses
        if not reinforce_only:
            candidates = list(active_cells - synapse_sources)
            if candidates:
                num_to_add = min(len(candidates), 
                               self.max_new_synapse_count - len(segment.synapses))
                if num_to_add > 0:
                    sample = np.random.choice(len(candidates), 
                                            size=min(num_to_add, self.sample_size),
                                            replace=False)
                    for idx in sample:
                        new_syn = DistalSynapse(candidates[idx], self.initial_permanence)
                        segment.synapses.append(new_syn)
    
    def reset(self) -> None:
        """Reset layer state."""
        self.active_cells.clear()
        self.winner_cells.clear()
        self.predictive_cells.clear()
        self.prev_active_cells.clear()
        self.prev_winner_cells.clear()
        self.active_columns.clear()
        for column in self.columns:
            column.active = False
            for cell in column.cells:
                cell.active = False
                cell.predictive = False
                cell.learning = False
    
    def get_cells(self) -> List[Cell]:
        """Return all cells in the layer."""
        cells = []
        for column in self.columns:
            cells.extend(column.cells)
        return cells
    
    def add_column(self) -> Column:
        """Add a new column to the layer."""
        col_idx = len(self.columns)
        new_column = Column([], position=(col_idx, 0), cells_per_column=self.cells_per_column)
        self.columns.append(new_column)
        return new_column


# ===== Spatial Pooler Layer =====

class SpatialPoolerLayer(Layer):
    """Spatial Pooler layer implementation following HTM principles.
    
    Learns sparse distributed representations of input patterns.
    """
    
    def __init__(self,
                 input_size: int,
                 num_columns: int,
                 potential_pct: float = 0.5,
                 sparsity: float = 0.02,
                 name: str = "SpatialPooler",
                 learning_rate: float = 1.0,
                 connected_perm: float = CONNECTED_PERM,
                 initial_permanence: float = INITIAL_PERMANENCE,
                 min_overlap: int = MIN_OVERLAP,
                 boost_strength: float = 0.0) -> None:
        super().__init__(name, learning_rate)
        self.input_size = input_size
        self.num_columns = num_columns
        self.potential_pct = potential_pct
        self.sparsity = sparsity
        self.connected_perm = connected_perm
        self.initial_permanence = initial_permanence
        self.min_overlap = min_overlap
        self.boost_strength = boost_strength
        
        # Create columns with proximal synapses
        self.columns: List[Column] = []
        grid_size = int(np.sqrt(num_columns))
        for i in range(num_columns):
            x = i % grid_size
            y = i // grid_size
            position = (x, y)
            
            # Create potential synapses
            num_potential = int(input_size * potential_pct)
            potential_inputs = np.random.choice(input_size, size=num_potential, replace=False)
            potential_synapses = [
                ProximalSynapse(int(idx), np.random.uniform(0, 1))
                for idx in potential_inputs
            ]
            
            col = Column(potential_synapses, position=position, cells_per_column=1)
            self.columns.append(col)
        
        self.input_vector: Optional[np.ndarray] = None
        
    def compute(self, learn: bool = True) -> None:
        """Compute spatial pooling."""
        if self.input_vector is None:
            return
        
        # Phase 1: Compute overlap
        for column in self.columns:
            column.compute_overlap(self.input_vector, self.min_overlap)
        
        # Phase 2: Inhibition
        active_columns = self._inhibit_columns()
        
        # Phase 3: Learning
        if learn:
            self._learn(active_columns)
        
        # Set active cells (one cell per active column in spatial pooler)
        new_active_cells: Set[Cell] = set()
        for col in active_columns:
            col.active = True
            new_active_cells.add(col.cells[0])
        
        self.set_active_cells(new_active_cells)
        self.winner_cells = new_active_cells.copy()
        
    def set_input(self, input_vector: np.ndarray) -> None:
        """Set input for spatial pooling."""
        self.input_vector = input_vector
        
    def _inhibit_columns(self) -> List[Column]:
        """Apply inhibition to select active columns."""
        # Global inhibition (simple version)
        num_active = max(1, int(self.num_columns * self.sparsity))
        sorted_columns = sorted(self.columns, key=lambda c: c.overlap, reverse=True)
        
        active_columns = []
        for col in sorted_columns[:num_active]:
            if col.overlap > 0:
                active_columns.append(col)
        
        return active_columns
    
    def _learn(self, active_columns: List[Column]) -> None:
        """Learn on active columns."""
        for column in active_columns:
            for syn in column.potential_synapses:
                if syn.source_input < len(self.input_vector):
                    if self.input_vector[syn.source_input]:
                        self._adjust_permanence(syn, increase=True)
                    else:
                        self._adjust_permanence(syn, increase=False)
            
            # Update connected synapses
            column._update_connected_synapses(self.connected_perm)
    
    def reset(self) -> None:
        """Reset layer state."""
        self.active_cells.clear()
        self.winner_cells.clear()
        self.input_vector = None
        for column in self.columns:
            column.active = False
            column.overlap = 0.0
            for cell in column.cells:
                cell.active = False
    
    def get_cells(self) -> List[Cell]:
        """Return all cells in the layer."""
        cells = []
        for column in self.columns:
            cells.extend(column.cells)
        return cells
    
    def add_column(self) -> Column:
        """Add a new column to the layer."""
        col_idx = len(self.columns)
        grid_size = int(np.sqrt(self.num_columns + 1))
        x = col_idx % grid_size
        y = col_idx // grid_size
        position = (x, y)
        
        # Create potential synapses for new column
        num_potential = int(self.input_size * self.potential_pct)
        potential_inputs = np.random.choice(self.input_size, size=num_potential, replace=False)
        potential_synapses = [
            ProximalSynapse(int(idx), np.random.uniform(0, 1))
            for idx in potential_inputs
        ]
        
        new_column = Column(potential_synapses, position=position, cells_per_column=1)
        self.columns.append(new_column)
        self.num_columns += 1
        return new_column


# ===== Custom Distal Layer (Fire-Together-Wire-Together) =====

class CustomDistalLayer(Layer):
    """Custom layer with distal segments that learn via fire-together-wire-together.
    
    Cells have distal segments that synapse onto another layer. Learning occurs
    through Hebbian-style plasticity: if a cell is active and has an active distal
    segment, strengthen that segment. Otherwise, create a new segment.
    """
    
    def __init__(self,
                 num_cells: int,
                 name: str = "CustomDistal",
                 learning_rate: float = 1.0,
                 activation_threshold: int = ACTIVATION_THRESHOLD,
                 initial_permanence: float = INITIAL_PERMANENCE,
                 sample_size: int = 20) -> None:
        super().__init__(name, learning_rate)
        self.num_cells = num_cells
        self.activation_threshold = activation_threshold
        self.initial_permanence = initial_permanence
        self.sample_size = sample_size
        
        # Create cells
        self.cells_list: List[Cell] = [Cell() for _ in range(num_cells)]
        
    def compute(self, learn: bool = True) -> None:
        """Compute activations and learn via fire-together-wire-together."""
        if not self.input_layers:
            return
        
        # Get input layer's active cells
        input_active_cells = set()
        for input_layer in self.input_layers:
            input_active_cells.update(input_layer.get_active_cells())
        
        new_active_cells: Set[Cell] = set()
        
        # Fire-together-wire-together: cells with active segments become active
        for cell in self.cells_list:
            has_active_segment = False
            
            for segment in cell.segments:
                active_syns = segment.active_synapses(input_active_cells)
                if len(active_syns) >= self.activation_threshold:
                    has_active_segment = True
                    new_active_cells.add(cell)
                    
                    if learn:
                        # Strengthen this segment (fire together, wire together)
                        self._strengthen_segment(segment, input_active_cells)
                    break
            
            # If cell is active but had no active segment, create new segment
            if cell.active and not has_active_segment and learn:
                self._create_new_segment(cell, input_active_cells)
        
        self.set_active_cells(new_active_cells)
        self.winner_cells = new_active_cells.copy()
    
    def _strengthen_segment(self, segment: Segment, active_cells: Set[Cell]) -> None:
        """Strengthen synapses to active cells in segment."""
        for syn in segment.synapses:
            if syn.source_cell in active_cells:
                self._adjust_permanence(syn, increase=True)
    
    def _create_new_segment(self, cell: Cell, active_cells: Set[Cell]) -> None:
        """Create new segment with synapses to sampled active cells."""
        if not active_cells:
            return
        
        new_segment = Segment()
        
        # Sample from active cells
        active_list = list(active_cells)
        num_to_sample = min(len(active_list), self.sample_size)
        sampled_cells = np.random.choice(len(active_list), size=num_to_sample, replace=False)
        
        for idx in sampled_cells:
            syn = DistalSynapse(active_list[idx], self.initial_permanence)
            new_segment.synapses.append(syn)
        
        cell.segments.append(new_segment)
    
    def reset(self) -> None:
        """Reset layer state."""
        self.active_cells.clear()
        self.winner_cells.clear()
        for cell in self.cells_list:
            cell.active = False
    
    def get_cells(self) -> List[Cell]:
        """Return all cells in the layer."""
        return self.cells_list.copy()
    
    def add_cell(self) -> Cell:
        """Add a new cell to the layer."""
        new_cell = Cell()
        self.cells_list.append(new_cell)
        self.num_cells += 1
        return new_cell


# ===== Sparsey-Inspired Spatial Pooler =====

class SparseyInspiredSpatialPooler(Layer):
    """Sparsey-inspired spatial pooler with neighborhood constraints.
    
    Groups of columns define neighborhoods, and only a fixed percentage
    of columns within each neighborhood can be active at any time.
    """
    
    def __init__(self,
                 input_size: int,
                 num_neighborhoods: int,
                 columns_per_neighborhood: int,
                 active_pct_per_neighborhood: float = 0.1,
                 potential_pct: float = 0.5,
                 name: str = "SparseyPooler",
                 learning_rate: float = 1.0,
                 connected_perm: float = CONNECTED_PERM,
                 initial_permanence: float = INITIAL_PERMANENCE,
                 min_overlap: int = MIN_OVERLAP) -> None:
        super().__init__(name, learning_rate)
        self.input_size = input_size
        self.num_neighborhoods = num_neighborhoods
        self.columns_per_neighborhood = columns_per_neighborhood
        self.active_pct_per_neighborhood = active_pct_per_neighborhood
        self.potential_pct = potential_pct
        self.connected_perm = connected_perm
        self.initial_permanence = initial_permanence
        self.min_overlap = min_overlap
        
        # Create neighborhoods of columns
        self.neighborhoods: List[List[Column]] = []
        total_columns = num_neighborhoods * columns_per_neighborhood
        
        for n in range(num_neighborhoods):
            neighborhood = []
            for c in range(columns_per_neighborhood):
                col_idx = n * columns_per_neighborhood + c
                position = (n, c)
                
                # Create potential synapses
                num_potential = int(input_size * potential_pct)
                potential_inputs = np.random.choice(input_size, size=num_potential, replace=False)
                potential_synapses = [
                    ProximalSynapse(int(idx), np.random.uniform(0, 1))
                    for idx in potential_inputs
                ]
                
                col = Column(potential_synapses, position=position, cells_per_column=1)
                neighborhood.append(col)
            
            self.neighborhoods.append(neighborhood)
        
        self.input_vector: Optional[np.ndarray] = None
        
    def compute(self, learn: bool = True) -> None:
        """Compute spatial pooling with neighborhood constraints."""
        if self.input_vector is None:
            return
        
        # Compute overlap for all columns
        for neighborhood in self.neighborhoods:
            for column in neighborhood:
                column.compute_overlap(self.input_vector, self.min_overlap)
        
        # Inhibition within each neighborhood
        active_columns = self._inhibit_neighborhoods()
        
        # Learning
        if learn:
            self._learn(active_columns)
        
        # Set active cells
        new_active_cells: Set[Cell] = set()
        for col in active_columns:
            col.active = True
            new_active_cells.add(col.cells[0])
        
        self.set_active_cells(new_active_cells)
        self.winner_cells = new_active_cells.copy()
    
    def set_input(self, input_vector: np.ndarray) -> None:
        """Set input for spatial pooling."""
        self.input_vector = input_vector
    
    def _inhibit_neighborhoods(self) -> List[Column]:
        """Apply inhibition within each neighborhood."""
        active_columns = []
        num_active_per_neighborhood = max(1, int(self.columns_per_neighborhood * 
                                                  self.active_pct_per_neighborhood))
        
        for neighborhood in self.neighborhoods:
            # Sort by overlap and pick top k
            sorted_cols = sorted(neighborhood, key=lambda c: c.overlap, reverse=True)
            
            for col in sorted_cols[:num_active_per_neighborhood]:
                if col.overlap > 0:
                    active_columns.append(col)
        
        return active_columns
    
    def _learn(self, active_columns: List[Column]) -> None:
        """Learn on active columns."""
        for column in active_columns:
            for syn in column.potential_synapses:
                if syn.source_input < len(self.input_vector):
                    if self.input_vector[syn.source_input]:
                        self._adjust_permanence(syn, increase=True)
                    else:
                        self._adjust_permanence(syn, increase=False)
            
            # Update connected synapses
            column._update_connected_synapses(self.connected_perm)
    
    def reset(self) -> None:
        """Reset layer state."""
        self.active_cells.clear()
        self.winner_cells.clear()
        self.input_vector = None
        for neighborhood in self.neighborhoods:
            for column in neighborhood:
                column.active = False
                column.overlap = 0.0
                for cell in column.cells:
                    cell.active = False
    
    def get_cells(self) -> List[Cell]:
        """Return all cells in the layer."""
        cells = []
        for neighborhood in self.neighborhoods:
            for column in neighborhood:
                cells.extend(column.cells)
        return cells
    
    def add_neighborhood(self) -> List[Column]:
        """Add a new neighborhood to the layer."""
        new_neighborhood = []
        n = len(self.neighborhoods)
        
        for c in range(self.columns_per_neighborhood):
            position = (n, c)
            
            # Create potential synapses
            num_potential = int(self.input_size * self.potential_pct)
            potential_inputs = np.random.choice(self.input_size, size=num_potential, replace=False)
            potential_synapses = [
                ProximalSynapse(int(idx), np.random.uniform(0, 1))
                for idx in potential_inputs
            ]
            
            col = Column(potential_synapses, position=position, cells_per_column=1)
            new_neighborhood.append(col)
        
        self.neighborhoods.append(new_neighborhood)
        self.num_neighborhoods += 1
        return new_neighborhood
