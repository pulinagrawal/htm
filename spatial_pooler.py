from sre_constants import IN
from turtle import pos
import numpy as np

from typing import (
    List,
    Set,
    Dict,
    Tuple,
    Optional,
    Sequence,
    Union,
)

CONNECTED_PERM = 0.5  # Permanence threshold for an input synapse to be considered connected
MIN_OVERLAP = 3       # Minimum overlap to be considered during inhibition
DESIRED_LOCAL_SPARSITY = 0.02  # Desired local sparsity for inhibition
INHIBITION_RADIUS = 1.0  # Sets the locality radius for desired sparsity

def euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate the Euclidean distance between two points."""
    return float(np.linalg.norm(np.array(pos1) - np.array(pos2)))

# Synapse class represents a synapse which connects a segment to a source input
class ProximalSynapse:
    """Proximal synapse (input space) used by Spatial Pooler only."""

    source_input: int
    permanence: float

    def __init__(self, source_input: int, permanence: float) -> None:
        self.source_input = source_input
        self.permanence = permanence

# Column class represents a column in the HTM Region
class Column:
    position: Tuple[int, int]
    potential_synapses: List[ProximalSynapse]
    boost: float
    active_duty_cycle: float
    overlap_duty_cycle: float
    min_duty_cycle: float
    connected_synapses: List[ProximalSynapse]
    overlap: float
    active: bool

    def __init__(self, potential_synapses: List[ProximalSynapse], position: Tuple[int, int] = (0, 0)) -> None:
        self.position = position
        self.potential_synapses = potential_synapses
        self.boost = 1.0
        self.active_duty_cycle = 0.0
        self.overlap_duty_cycle = 0.0
        self.min_duty_cycle = 0.01
        self.connected_synapses = [s for s in potential_synapses if s.permanence > CONNECTED_PERM]
        self.overlap = 0.0
        self.active = False

    def compute_overlap(self, input_vector: np.ndarray) -> None:
        """Compute overlap with current binary input vector and apply boost."""
        overlap = sum(1 for s in self.connected_synapses if input_vector[s.source_input])
        self.overlap = float(overlap * self.boost) if overlap >= MIN_OVERLAP else 0.0

class SpatialPooler:
    columns: List[Column]
    num_columns: int
    input_size: int

    # Input variants accepted by compute_active_columns
    ColumnField = Union[np.ndarray,
                        Sequence[Column], 
                        Sequence[int]]
    InputField = Union[np.ndarray, Sequence[int]]
    InputComposite = Union[
        np.ndarray,
        Sequence[int],
        Sequence[InputField],
        Dict[str, InputField],
    ]
    # Add random activations of columns instead of dutyycle boosting for simplicity

    def __init__(self, num_columns: int, input_size: int, potential_receptive_field_pct: float = 0.5) -> None:
        self.num_columns = num_columns
        self.input_size = input_size
        self.columns = self._initialize_columns(num_columns, input_size, potential_receptive_field_pct)

    def _initialize_columns(self, num_columns: int, input_size: int, potential_receptive_field_pct: float) -> List[Column]:
        columns = []
        for i in range(num_columns):
            potential_synapses = []
            num_potential_synapses = int(input_size * potential_receptive_field_pct)
            potential_indices = np.random.choice(input_size, num_potential_synapses, replace=False)
            for idx in potential_indices:
                permanence = np.random.rand()
                potential_synapses.append(ProximalSynapse(source_input=idx, permanence=permanence))
            columns.append(Column(potential_synapses=potential_synapses))
        return columns
      
    def set_active_columns(self, column_state: ColumnField) -> None:
        """Compute and return the list of active columns for the given input vector."""
        if (isinstance(column_state, (list, tuple)) and all(isinstance(c, int) for c in column_state)) \
            or isinstance(column_state, np.ndarray):
            # This is a list/array of column indices
            if len(column_state) != len(self.columns):
                raise ValueError("Length of column_state does not match number of columns.")

            for col, state in zip(self.columns, column_state):
                col.active = bool(state)
        elif isinstance(column_state, (list, tuple)) and all(isinstance(c, Column) for c in column_state):
            raise NotImplementedError("Setting active columns from Column objects is not implemented.")
    
    def run(self, input_vector: InputComposite, learn: bool=True) -> List[Column]:
        """Compute active columns for the given input vector and apply inhibition."""
        combined = self.combine_input_fields(input_vector)
        for column in self.columns:
            column.compute_overlap(combined)
        active_columns = self.inhibition(self.columns)
        # Set active state
        for col in self.columns:
            col.active = col in active_columns
        # Store current input indices for learning
        self.current_input_indices = set(np.where(combined > 0)[0])
        return active_columns

    def learn(self, learning_rate = 0.1) -> None:
        """Update synapse permanences based on active columns and input vector."""
        for col in self.columns:
            if col.active:
                for syn in col.potential_synapses:
                    if syn.source_input in self.current_input_indices:
                        syn.permanence = min(1.0, syn.permanence + learning_rate)  # Increase permanence
                    else:
                        syn.permanence = max(0.0, syn.permanence - learning_rate)  # Decrease permanence
                # Update connected synapses
                col.connected_synapses = [s for s in col.potential_synapses if s.permanence > CONNECTED_PERM]

    # Computes the active columns after applying inhibition
    def _compute_active_columns_list(
        self,
        input_vector: InputComposite,
        inhibition_radius: float,
    ) -> List[Column]:
        """Internal: return list of active Column objects (legacy behavior)."""
        combined = self.combine_input_fields(input_vector)
        for column in self.columns:
            column.compute_overlap(combined)
        return self.inhibition(self.columns, inhibition_radius)

    # Applies inhibition to determine which columns will become active
    def inhibition(self, columns: Sequence[Column], inhibition_radius: float=INHIBITION_RADIUS) -> List[Column]:
        active_columns: List[Column] = []
        
        def column_neighbors(c: Column) -> List[Column]:
            return [c2 for c2 in columns  
                    if c != c2 and euclidean_distance(c.position, c2.position) <= inhibition_radius]
        
        for c in columns:
        # Find neighbors of column c
            neighbors = column_neighbors(c)
            min_local_activity = self.kth_score(neighbors, DESIRED_LOCAL_SPARSITY)
            if c.overlap > 0 and c.overlap >= min_local_activity:
                active_columns.append(c)
        return active_columns

    # Returns the k-th highest overlap value from a list of columns
    def kth_score(self, neighbors: Sequence[Column], sparsity: float) -> float:
        k = int(np.ceil(sparsity * len(neighbors)))
        if not neighbors:
            return 0
        ordered = sorted(neighbors, key=lambda x: x.overlap, reverse=True)
        if k <= 0:
            return 0
        if k > len(ordered):
            # If fewer neighbors than desired local activity, use lowest neighbor overlap
            return float(ordered[-1].overlap) if ordered else 0.0
        return float(ordered[k-1].overlap)


    def combine_input_fields(self, input_vector: InputComposite) -> np.ndarray:
      """Prepare / combine input fields into a single binary numpy array.

      Responsibilities moved out of `compute_active_columns` so they can be used
      independently by the new single-step `run` helper or external callers.

      Accepted forms:
        - 1D array-like of ints (already concatenated)
        - list / tuple of field arrays ⇒ concatenated
        - dict[str, field array] ⇒ concatenated (order preserved) and field
          metadata (`field_ranges`, `field_order`, `column_field_map` reset)

      Returns: np.ndarray (dtype=int) of length == `input_space_size`.
      Raises: ValueError if combined length mismatches configured size.
      """
      # Dict branch (ordered by insertion order)
      if isinstance(input_vector, dict):
          start = 0
          arrays = []
          self.field_ranges = {}
          self.field_order = []
          for name, arr in input_vector.items():
              a = np.asarray(arr, dtype=int)
              end = start + a.shape[0]
              self.field_ranges[name] = (start, end)
              self.field_order.append(name)
              arrays.append(a)
              start = end
          combined = np.concatenate(arrays) if arrays else np.array([], dtype=int)
          # Invalidate previous mapping; will be recomputed lazily
          self.column_field_map = {}
      elif isinstance(input_vector, (list, tuple)):
          arrays = [np.asarray(v, dtype=int) for v in input_vector]
          combined = np.concatenate(arrays) if arrays else np.array([], dtype=int)
      else:
          combined = np.asarray(input_vector, dtype=int)

      if combined.shape[0] != self.input_size:
          raise ValueError(
              f"Combined input length {combined.shape[0]} != configured input_space_size {self.input_size}."
          )

      # Defensive structure check for previously stored metadata
      if self.field_ranges and any(len(rg) != 2 for rg in self.field_ranges.values()):
          self.field_ranges = {}
          self.field_order = []
          self.column_field_map = {}

      # Build mapping if new metadata present
      if self.field_ranges and not self.column_field_map:
          self._assign_column_fields()
      return combined
    
    def _assign_column_fields(self) -> None:
      """Assign each column a dominant field based on connected synapse source indices.
      Dominant field = field with largest count of connected synapse inputs.
      Ties broken by earliest field order. Stores mapping in self.column_field_map."""
      if not self.field_ranges:
          return
      inv_order = {name: i for i, name in enumerate(self.field_order)}
      for col in self.columns:
          counts = {}
          for syn in col.connected_synapses:
              idx = syn.source_input
              for name, (s, e) in self.field_ranges.items():
                  if s <= idx < e:
                      counts[name] = counts.get(name, 0) + 1
                      break
          if counts:
              # select max count, break ties by field order
              best = sorted(counts.items(), key=lambda kv: (-kv[1], inv_order[kv[0]]))[0][0]
              self.column_field_map[col] = best
          else:
              self.column_field_map[col] = None