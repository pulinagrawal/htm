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
PERMANENCE_INC = 0.05  # Amount by which synapses are incremented during learning
PERMANENCE_DEC = 0.05  # Amount by which synapses are decremented during learning

InputField = Union[np.ndarray, Sequence[int]]
InputComposite = Union[
    np.ndarray,
    Sequence[int],
    Sequence[InputField],
    Dict[str, InputField],
]
ActiveColumnInput = Union[Set[int], Sequence[int], np.ndarray]


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

    def find_best_segment(self, active_cells: Set['Cell']) -> Tuple[Optional['Segment'], int]:
        """Return segment with most synapses to the given active cells.

        The tuple structure avoids call-site conditionals and always exposes both
        the winning segment (if any) and its match count.
        """
        best_segment = None
        max_synapses_to_active_cells = -1
        
        for segment in self.segments:
            synapses_to_active_cells = segment.get_synapses_to_active_cells(active_cells)
            if len(synapses_to_active_cells) > max_synapses_to_active_cells:
                max_synapses_to_active_cells = len(synapses_to_active_cells)
                best_segment = segment

        return best_segment, max_synapses_to_active_cells

class Synapse:
    
    def __init__(self, source_cell: Cell|None, permanence: float) -> None:
        self.source_cell: Cell|None = source_cell
        self.permanence: float = permanence

    def _adjust_permanence(self, increase: bool, strength: float=1.0) -> None:
        """Adjust synapse permanence by learning rate."""
        if increase:
            self.permanence = min(1.0, self.permanence + PERMANENCE_INC * strength)
        else:
            self.permanence = max(0.0, self.permanence - PERMANENCE_DEC * strength)

class DistalSynapse(Synapse):
    """Distal synapse connecting to a source cell."""
    
    def __init__(self, source_cell: Cell, permanence: float) -> None:
        super().__init__(source_cell, permanence)

class Segment:
    """Distal segment composed of synapses to cells."""
    
    def __init__(
        self,
        synapses: Optional[List[DistalSynapse]] = None,
        activation_threshold: Optional[int] = None,
    ) -> None:
        self.synapses: List[DistalSynapse] = synapses if synapses is not None else []
        self.sequence_segment: bool = False  # True if learned in a predictive context
        self.activation_threshold: int = (
            activation_threshold if activation_threshold is not None else ACTIVATION_THRESHOLD
        )
    
    def active_synapses(self, active_cells: Set[Cell], connected_perm: float = CONNECTED_PERM) -> List[DistalSynapse]:
        """Return connected synapses whose source cell is active."""
        return [syn for syn in self.synapses 
                if syn.source_cell in active_cells and syn.permanence >= connected_perm]
    
    def get_synapses_to_active_cells(self, active_cells: Set[Cell]) -> List[DistalSynapse]:
        """Return synapses whose source cell is active (ignores permanence threshold)."""
        return [syn for syn in self.synapses if syn.source_cell in active_cells]
    
    def meets_activation_threshold(
        self,
        active_cells: Set[Cell],
        threshold: Optional[int] = None,
        connected_perm: float = CONNECTED_PERM,
    ) -> bool:
        """Return True if segment has at least threshold connected synapses to active cells."""
        effective_threshold = self.activation_threshold if threshold is None else threshold
        if effective_threshold <= 0:
            return True
        count = 0
        for syn in self.synapses:
            if syn.permanence >= connected_perm and syn.source_cell in active_cells:
                count += 1
                if count >= effective_threshold:
                    return True
        return False


class ProximalSynapse(Synapse):
    """Proximal synapse connecting to an input bit."""
    def __init__(self, source_input: int, permanence: float) -> None:
        super().__init__(source_cell=None, permanence=permanence)
        self.source_input: int = source_input  # Index of input bit this synapse connects to


class Column:
    """Column containing cells and proximal synapses for spatial pooling."""
    
    def __init__(self, 
                 potential_synapses: List[ProximalSynapse],
                 position: Optional[Tuple[int, int]] = None,
                 cells_per_column: int = 1) -> None:
        self.potential_synapses: List[ProximalSynapse] = potential_synapses
        self.position: Optional[Tuple[int, int]] = position
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
        """Compute overlap with current binary input vector."""
        overlap = sum(1 for s in self.connected_synapses 
                     if s.source_input < len(input_vector) and input_vector[s.source_input])
        self.overlap = float(overlap) if overlap >= min_overlap else 0.0
    
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
        
    @abstractmethod
    def compute(
        self,
        input: Optional[ActiveColumnInput] = None,
        learn: bool = True,
    ) -> None:
        """Compute layer activations given new input or previously set state."""
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


class ColumnarLayer(Layer):
    """Layer variant that manages spatial columns and field metadata."""

    def __init__(self, name: str = "Layer", learning_rate: float = 1.0) -> None:
        super().__init__(name, learning_rate)
        self.columns: List[Column] = []
        self.field_ranges: Dict[str, Tuple[int, int]] = {}
        self.field_order: List[str] = []
        self.column_field_map: Dict[Column, Optional[str]] = {}

    def combine_input_fields(self, input_vector: InputComposite, expected_size: Optional[int] = None) -> np.ndarray:
        """Combine multi-field inputs into a single binary vector like HTM TemporalPooler."""
        combined, metadata = self._combine_input_fields(input_vector)

        if expected_size is not None and combined.shape[0] != expected_size:
            raise ValueError(
                f"Combined input length {combined.shape[0]} != expected size {expected_size}."
            )

        if metadata is not None:
            self.field_ranges, self.field_order = metadata
            self.column_field_map = {}
            self._assign_column_fields()
        else:
            self.field_ranges = {}
            self.field_order = []
            self.column_field_map = {}

        return combined

    def _combine_input_fields(
        self,
        input_vector: InputComposite,
    ) -> Tuple[np.ndarray, Optional[Tuple[Dict[str, Tuple[int, int]], List[str]]]]:
        """Pure helper that concatenates fields and reports metadata when present."""
        if isinstance(input_vector, dict):
            start = 0
            arrays: List[np.ndarray] = []
            field_ranges: Dict[str, Tuple[int, int]] = {}
            field_order: List[str] = []
            for name, arr in input_vector.items():
                arr_np = np.asarray(arr, dtype=int).ravel()
                end = start + arr_np.shape[0]
                field_ranges[name] = (start, end)
                field_order.append(name)
                arrays.append(arr_np)
                start = end
            combined = np.concatenate(arrays) if arrays else np.array([], dtype=int)
            metadata = (field_ranges, field_order)
        elif isinstance(input_vector, (list, tuple)):
            arrays = [np.asarray(v, dtype=int).ravel() for v in input_vector]
            combined = np.concatenate(arrays) if arrays else np.array([], dtype=int)
            metadata = None
        else:
            combined = np.asarray(input_vector, dtype=int).ravel()
            metadata = None

        return combined, metadata

    def _assign_column_fields(self) -> None:
        """Assign dominant field per column based on connected synapse sources."""
        if not self.field_ranges or not self.columns:
            return
        inv_order = {name: idx for idx, name in enumerate(self.field_order)}
        for column in self.columns:
            counts: Dict[str, int] = {}
            for syn in getattr(column, 'connected_synapses', []):
                idx = getattr(syn, 'source_input', None)
                if idx is None:
                    continue
                for field_name, (start, end) in self.field_ranges.items():
                    if start <= idx < end:
                        counts[field_name] = counts.get(field_name, 0) + 1
                        break
            if counts:
                best = sorted(
                    counts.items(),
                    key=lambda kv: (-kv[1], inv_order.get(kv[0], float('inf')))
                )[0][0]
                self.column_field_map[column] = best
            else:
                self.column_field_map[column] = None

    def _ensure_field_assignments(self) -> None:
        """Build column->field assignments when metadata exists but mapping is empty."""
        if self.field_ranges and not self.column_field_map:
            self._assign_column_fields()

    def split_indices_by_field(
        self,
        column_indices: Set[int],
        include_all_bucket: bool = True,
        unassigned_label: str = "__unassigned__",
    ) -> Dict[str, Set[int]]:
        """Group column indices by their associated input field."""
        if not column_indices:
            return {}

        if not self.columns:
            raise RuntimeError(
                f"{self.__class__.__name__} cannot group columns without initialized columns."
            )

        indexed_columns = {idx: col for idx, col in enumerate(self.columns)}
        self._ensure_field_assignments()

        buckets: Dict[str, Set[int]] = {}
        for idx in sorted(column_indices):
            field_name: Optional[str] = None
            column = indexed_columns.get(idx)
            if column is not None and self.column_field_map:
                field_name = self.column_field_map.get(column)

            if field_name is None and self.field_ranges:
                for name, (start, end) in self.field_ranges.items():
                    if start <= idx < end:
                        field_name = name
                        break

            bucket_name = field_name if field_name is not None else unassigned_label
            buckets.setdefault(bucket_name, set()).add(idx)

        if include_all_bucket:
            buckets["__all__"] = set(column_indices)

        return {key: value for key, value in buckets.items() if value}

# ===== Temporal Memory Layer =====

class TemporalMemoryLayer(ColumnarLayer):
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
                 segment_growth_speed: float = 1.0,
                 initial_permanence: float = INITIAL_PERMANENCE) -> None:
        super().__init__(name, learning_rate)
        self.num_columns = num_columns
        self.cells_per_column = cells_per_column
        self.activation_threshold = activation_threshold
        self.learning_threshold = learning_threshold
        self.max_new_synapse_count = max_new_synapse_count
        self.initial_permanence = initial_permanence
        self.segment_growth_speed = segment_growth_speed  # % of possible new synapses to add per growth
        
        # Create columns with cells
        self.columns: List[Column] = []
        for i in range(num_columns):
            col = Column([], position=(i, 0), cells_per_column=cells_per_column)
            self.columns.append(col)
        
        self.active_columns: Set[int] = set()
        self.bursting_columns: Set[int] = set()
        self.predictive_cells: Set[Cell] = set()
        self.prev_active_cells: Set[Cell] = set()
        
    def compute(
        self,
        input: InputComposite = None,
        learn: bool = True,
    ) -> None:
        """Compute temporal memory activations."""
        if input is not None:
            self.set_active_columns(self.encode_active_columns(input))

        # Phase 1: Activate cells
        self._activate_cells(learn)
        
        # Phase 2: Activate dendrites for next timestep
        self._activate_dendrites()
        
    def encode_active_columns(self, data: InputComposite) -> Set[int]:
        """Convert composite/binary inputs into the canonical active column set."""
        vector = self.combine_input_fields(data, expected_size=self.num_columns)
        return set(map(int, np.flatnonzero(vector)))

    def _validate_column_index(self, idx: Any) -> int:
        value = int(idx)
        if value < 0 or value >= self.num_columns:
            raise ValueError(f"Column index {value} out of bounds for {self.num_columns} columns.")
        return value

    def set_active_columns(self, active_columns: ActiveColumnInput) -> None:
        """Set which columns are active (typically from spatial pooler).
        Accepts either a set of indices, an explicit index sequence, or a binary mask.
        """
        if isinstance(active_columns, set):
            self.active_columns = {self._validate_column_index(idx) for idx in active_columns}
            return

        if isinstance(active_columns, np.ndarray):
            vector = np.asarray(active_columns).ravel()
            if vector.size == self.num_columns and np.all(np.logical_or(vector == 0, vector == 1)):
                self.active_columns = set(map(int, np.flatnonzero(vector)))
                return
            self.active_columns = {self._validate_column_index(val) for val in vector.tolist()}
            return

        if isinstance(active_columns, Sequence):
            if isinstance(active_columns, (str, bytes)):
                raise TypeError("Active column sequence must not be a string/bytes.")

            seq_len = len(active_columns)
            if seq_len == self.num_columns and all(value in (0, 1, False, True) for value in active_columns):
                indices: Set[int] = {self._validate_column_index(idx) for idx, value in enumerate(active_columns) if value}
                self.active_columns = indices
                return

            self.active_columns = {self._validate_column_index(value) for value in active_columns}
            return

        raise TypeError("Unsupported type for active_columns; provide indices or a binary mask.")
        
    def _activate_cells(self, learn: bool) -> None:
        """Activate cells based on active columns and predictions."""
        new_active_cells: Set[Cell] = set()
        new_bursting_columns: Set[int] = set()
        predicted_learning_segments: List[Segment] = []
        bursting_winners: List[Cell] = []
        
        for col_idx in self.active_columns:
            if col_idx >= len(self.columns):
                continue
            column = self.columns[col_idx]
            
            # Check for predicted cells in this column
            predicted_cells = [cell for cell in column.cells if cell in self.predictive_cells]
            
            if predicted_cells:
                # Activate predicted cells
                for cell in predicted_cells:
                    # TODO: Add ability to inhibit to maintain sparsity
                    new_active_cells.add(cell)
                    
                    if learn:
                        for segment in cell.segments:
                            if segment.meets_activation_threshold(self.prev_active_cells):
                                predicted_learning_segments.append(segment)
            else:
                # Burst column - activate all cells
                new_bursting_columns.add(col_idx)
                for cell in column.cells:
                    new_active_cells.add(cell)
                
                if learn:
                    winner_cell = self._get_best_matching_cell(column, self.prev_active_cells)
                    bursting_winners.append(winner_cell)
        
        # Update state
        self.bursting_columns = new_bursting_columns

        
        # TODO: Possibly the learn in previous if-block and here can be combined to
        # simplify code 
        # TODO: More bursting more the learning_rate
        if learn:
            for cell in new_active_cells:
                for segment in cell.segments:
                    if segment.meets_activation_threshold(self.prev_active_cells):
                        self._strengthen_segment(segment, self.prev_active_cells)
            for winner in bursting_winners:
                self._learn_on_cell(winner, self.prev_active_cells)

        self.prev_active_cells = self.active_cells.copy()
        self.set_active_cells(new_active_cells)
        
    def _activate_dendrites(self) -> None:
        """Update predictive cells for next timestep."""
        new_predictive_cells: Set[Cell] = set()
        
        for column in self.columns:
            for cell in column.cells:
                for segment in cell.segments:
                    if segment.meets_activation_threshold(self.active_cells):
                        new_predictive_cells.add(cell)
                        break
        
        self.predictive_cells = new_predictive_cells
    
    def get_predicted_columns(self, separate_fields: bool = False) -> Union[Set[int], Dict[str, Set[int]]]:
        """Return predicted column indices, optionally grouped by field metadata."""
        predicted_columns: Set[int] = set()
        if not self.predictive_cells:
            return {} if separate_fields else predicted_columns

        for idx, column in enumerate(self.columns):
            if any(cell in self.predictive_cells for cell in column.cells):
                predicted_columns.add(idx)

        if not separate_fields:
            return predicted_columns

        return self.split_indices_by_field(predicted_columns)
        
    def _get_best_matching_cell(self, column: Column, cell_activations: Set[Cell]) -> Cell:
        """Select cell whose segment has most synapses to
           the input cell activations (typically prev active cells);
             fallback to cell with fewest segments.
        """
        best_cell = None
        highest_synapse_count = -1
        
        # Find a cell with segment that has most synapses to cell activations
        for cell in column.cells:
            _, max_synapses_to_active_cells  = cell.find_best_segment(cell_activations)
            if max_synapses_to_active_cells > highest_synapse_count:
                highest_synapse_count = max_synapses_to_active_cells
                best_cell = cell
        
        if best_cell is None:
            # Select cell with fewest segments 
            best_cell = min(column.cells, key=lambda c: len(c.segments))
        
        return best_cell
    

    def _learn_on_cell(self, cell: Cell, prev_active_cells: Set[Cell]) -> None:
        """Create or adapt segment on cell to learn pattern."""
        # Find segment with most synapses to the cell activations
        best_segment, synapses_to_active_cells = cell.find_best_segment(prev_active_cells)
        
        if best_segment is None or synapses_to_active_cells < self.learning_threshold:
            # Create new segment
            best_segment = Segment(activation_threshold=self.activation_threshold)
            cell.segments.append(best_segment)
        
        # Adapt segment
        self._strengthen_segment(best_segment, prev_active_cells)
        self._grow_segment(best_segment, prev_active_cells)
    
    def _strengthen_segment(self, segment: Segment, prev_active_cells: Set[Cell]) -> None:
        # Strengthen synapses to active cells
        for syn in segment.synapses:
            if syn.source_cell in prev_active_cells:
                syn._adjust_permanence(increase=True, strength=self.learning_rate)
            else:
                syn._adjust_permanence(increase=False, strength=self.learning_rate)

    def _grow_segment(self, segment: Segment, prev_active_cells: Set[Cell]) -> None:
        """Adapt segment synapses based on active cells."""
        # Get synapse sources
        source_cells = {syn.source_cell for syn in segment.synapses}
        
        # TODO: Extract this logic to a utility function
        
        # Grow new synapses on this segement to active cells not already connected
        # Find candidates for new synapses
        candidates = list(prev_active_cells - source_cells)
        available_synapses = self.max_new_synapse_count - len(segment.synapses)
        num_to_add = min(len(candidates), available_synapses)
        if num_to_add > 0:
            try:
                sample = np.random.choice(len(candidates), size=int(num_to_add*self.segment_growth_speed), replace=False)
            except ValueError:
                # print for debugging
                print("ValueError in np.random.choice: num_to_add =", num_to_add, "candidates len =", len(candidates), 
                      "segment synapses len =", len(segment.synapses), "segment_growth_speed =", self.segment_growth_speed)
                raise ValueError("Error in np.random.choice for growing segment synapses.")
            for idx in sample:
                new_syn = DistalSynapse(candidates[idx], self.initial_permanence)
                segment.synapses.append(new_syn)
    
    def reset(self) -> None:
        """Reset layer state."""
        self.active_cells.clear()
        self.predictive_cells.clear()
        self.prev_active_cells.clear()
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

class SpatialPoolerLayer(ColumnarLayer):
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
                 min_overlap: int = MIN_OVERLAP) -> None:
        super().__init__(name, learning_rate)
        self.input_size = input_size
        self.num_columns = num_columns
        self.potential_pct = potential_pct
        self.sparsity = sparsity
        self.connected_perm = connected_perm
        self.initial_permanence = initial_permanence
        self.min_overlap = min_overlap
        
        # Create columns with proximal synapses
        self.columns: List[Column] = []
        for col_idx in range(num_columns):
            
            # Create potential synapses
            num_potential = int(input_size * potential_pct)
            potential_inputs = np.random.choice(input_size, size=num_potential, replace=False)
            potential_synapses = [
                ProximalSynapse(int(idx), np.random.uniform(0, 1))
                for idx in potential_inputs
            ]
            
            col = Column(potential_synapses, cells_per_column=1)
            self.columns.append(col)
        
        self.input_vector: Optional[np.ndarray] = None
        
    def compute(
        self,
        input: Optional[Union[Set[int], Sequence[int], np.ndarray, InputComposite]] = None,
        learn: bool = True,
    ) -> None:
        """Compute spatial pooling."""
        if input is not None:
            self.set_input(input)

        if self.input_vector is None:
            raise ValueError(
                "SpatialPoolerLayer.compute() requires input or a prior set_input() call."
            )
        
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
        
    def set_input(self, input_vector: Union[Set[int], Sequence[int], np.ndarray, InputComposite]) -> None:
        """Set input for spatial pooling."""
        self.input_vector = self.combine_input_fields(input_vector, expected_size=self.input_size)
        
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
                        syn._adjust_permanence(increase=True)
                    else:
                        syn._adjust_permanence(increase=False)
            
            # Update connected synapses
            column._update_connected_synapses(self.connected_perm)
    
    def reset(self) -> None:
        """Reset layer state."""
        self.active_cells.clear()
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
        # Create potential synapses for new column
        num_potential = int(self.input_size * self.potential_pct)
        potential_inputs = np.random.choice(self.input_size, size=num_potential, replace=False)
        potential_synapses = [
            ProximalSynapse(int(idx), np.random.uniform(0, 1))
            for idx in potential_inputs
        ]
        
        new_column = Column(potential_synapses, cells_per_column=1)
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
        
    def compute(
        self,
        input: Optional[Union[Set[int], Sequence[int], np.ndarray, InputComposite]] = None,
        learn: bool = True,
    ) -> None:
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
                if segment.meets_activation_threshold(input_active_cells):
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
    
    def _strengthen_segment(self, segment: Segment, active_cells: Set[Cell]) -> None:
        """Strengthen synapses to active cells in segment."""
        for syn in segment.synapses:
            if syn.source_cell in active_cells:
                self._adjust_permanence(syn, increase=True)
    
    def _create_new_segment(self, cell: Cell, active_cells: Set[Cell]) -> None:
        """Create new segment with synapses to sampled active cells."""
        if not active_cells:
            return
        
        new_segment = Segment(activation_threshold=self.activation_threshold)
        
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

class SparseyInspiredSpatialPooler(ColumnarLayer):
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
        self.columns = [column for neighborhood in self.neighborhoods for column in neighborhood]
        
        self.input_vector: Optional[np.ndarray] = None
        
    def compute(
        self,
        input: Optional[Union[Set[int], Sequence[int], np.ndarray, InputComposite]] = None,
        learn: bool = True,
    ) -> None:
        """Compute spatial pooling with neighborhood constraints."""
        if input is not None:
            self.set_input(input)

        if self.input_vector is None:
            raise ValueError(
                "SparseyInspiredSpatialPooler.compute() requires input or a prior set_input() call."
            )
        
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
    
    def set_input(self, input_vector: Union[Set[int], Sequence[int], np.ndarray, InputComposite]) -> None:
        """Set input for spatial pooling."""
        self.input_vector = self.combine_input_fields(input_vector, expected_size=self.input_size)
    
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
                        syn._adjust_permanence(increase=True)
                    else:
                        syn._adjust_permanence(increase=False)
            
            # Update connected synapses
            column._update_connected_synapses(self.connected_perm)
    
    def reset(self) -> None:
        """Reset layer state."""
        self.active_cells.clear()
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
        self.columns.extend(new_neighborhood)
        self.num_neighborhoods += 1
        return new_neighborhood
