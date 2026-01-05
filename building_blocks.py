from itertools import chain
import random
from re import I
from typing import (
    Iterable,
    List,
    Set,
    Tuple,
    Optional,
)

from statistics import fmean, pstdev

from rdse import RDSEParameters, RandomDistributedScalarEncoder

# Constants
CONNECTED_PERM = 0.5  # Permanence threshold for a synapse to be considered connected
DESIRED_LOCAL_SPARSITY = 0.02  # Desired local sparsity for inhibition
INITIAL_PERMANENCE = 0.21  # Initial permanence for new synapses
PERMANENCE_INC = 0.05  # Amount by which synapses are incremented during learning
PERMANENCE_DEC = 0.05  # Amount by which synapses are decremented during learning
GROWTH_STRENGTH = 0.1  # Fraction of max synapses to grow on a segment during learning
RECEPTIVE_FIELD_PCT = 0.2 # Percentage of distal field sampled by a segment for potential synapses


def make_state_class(label: str):
    """Create a mixin that tracks current and previous boolean states for `label`."""
    attr = label.lower()
    prev_attr = f"prev_{attr}"
    new_class = None

    def __init__(self, *args, **kwargs):
        super(new_class, self).__init__(*args, **kwargs)
        setattr(self, attr, getattr(self, attr, False))
        setattr(self, prev_attr, getattr(self, prev_attr, False))

    def set_state(self):
        setattr(self, attr, True)

    def clear_state(self):
        setattr(self, attr, False)

    def update_prev_state(self):
        setattr(self, prev_attr, getattr(self, attr))
        setattr(self, attr, False)

    namespace = {
        "__init__": __init__,
        "state_name": attr,
        "prev_state_name": prev_attr,
        f"set_{attr}": set_state,
        f"clear_{attr}": clear_state,
        f"update_prev_{attr}": update_prev_state,
    }

    new_class = type(label.capitalize(), (object,), namespace)
    return new_class

Active = make_state_class("active")
Predictive = make_state_class("predictive")
Bursting = make_state_class("bursting")
Learning = make_state_class("learning")

# ===== Basic Building Blocks =====


class Cell(Active, Predictive):
    """Single cell within a column or layer.
    
    Holds a (possibly empty) list of distal segments used for temporal learning.
    """
    
    def __init__(
        self,
        parent_column: 'Column' = None,
        distal_field: 'Field' = None,
    ) -> None:
        super().__init__()
        self.parent_column = parent_column
        self.distal_field = distal_field
        self.segments: List['Segment'] = []
        
    def initialize(self, distal_field: 'Field') -> None:
        self.distal_field = distal_field
    
    def __repr__(self) -> str:
        return f"Cell(id={id(self)})"

    def find_best_predictive_segment(self) -> 'Segment':
        """Return segment with most synapses to the given active cells.

        The tuple structure avoids call-site conditionals and always exposes both
        the winning segment (if any) and its match count.
        """
        best_segment = None
        max_synapses_to_active_cells = -1
        
        for segment in self.segments:
            # Code can improved to push the learning threshold down to segment level
            synapses_to_active_cells = segment.get_synapses_to_prev_active_cells()
            if len(synapses_to_active_cells) > max_synapses_to_active_cells:
                max_synapses_to_active_cells = len(synapses_to_active_cells)
                best_segment = segment


        return best_segment

    def get_best_learning_segment(self):
        best_learning_segment = self.find_best_predictive_segment()
        if best_learning_segment is None or not best_learning_segment.meets_learning_threshold():
            best_learning_segment = Segment(
                parent_cell=self,
            )
            self.segments.append(best_learning_segment)
        return best_learning_segment

    def clear_states(self) -> None:
        self.update_prev_active()
        for segment in self.segments:
            segment.clear_states()



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

class Segment(Active, Learning):
    """Distal segment composed of synapses to cells."""
    
    def __init__(
        self,
        parent_cell: Cell = None,
        synapses: Optional[List[DistalSynapse]] = None,
    ) -> None:
        super().__init__()
        self.parent_cell: Cell = parent_cell
        self.synapses: List[DistalSynapse] = synapses if synapses is not None else []
        self.sequence_segment: bool = False  # True if learned in a predictive context
        self.max_synapses = int(.2*len(self.parent_cell.distal_field.cells))
        self.activation_threshold: int = int(.002*self.max_synapses)
        self.learning_threshold_connected_pct: float = .1
    
    def activate_segment(self) -> List[DistalSynapse]:
        """Return connected synapses whose source cell is active."""
        connected_synapses = [syn for syn in self.synapses 
                              if syn.source_cell.active and syn.permanence >= CONNECTED_PERM]
        if len(connected_synapses) >= self.activation_threshold:
            self.set_active()
            self.parent_cell.set_predictive()
    
    def meets_learning_threshold(self) -> bool:
        """Return True if segment meets learning threshold."""
        return len(self.get_synapses_to_prev_active_cells()) > self.learning_threshold_connected_pct*len(self.synapses):
    
    def get_synapses_to_prev_active_cells(self) -> List[DistalSynapse]:
        """Return synapses whose source cell is active (ignores permanence threshold)."""
        return [syn for syn in self.synapses if syn.source_cell in self.parent_cell.distal_field.prev_active_cells]
    
    def adapt(self, strength) -> None:
        # Strengthen synapses to active cells
        prev_active_cells = self.parent_cell.distal_field.prev_active_cells-{self.parent_cell}
        for syn in self.synapses:
            if syn.source_cell in prev_active_cells:
                syn._adjust_permanence(increase=True, strength=strength)
            else:
                syn._adjust_permanence(increase=False, strength=strength)
    
    def grow_synapses(self, strength) -> None:
        """Grow new synapses to random cells in the distal field."""
        potential_cells = list(self.parent_cell.distal_field.prev_active_cells - {syn.source_cell for syn in self.synapses} - {self.parent_cell})
        random.shuffle(potential_cells)
        available_synapses = self.max_synapses - len(self.synapses)
        cells_to_connect = potential_cells[:int(available_synapses * GROWTH_STRENGTH * strength)]
        
        for cell in cells_to_connect:
            new_syn = DistalSynapse(source_cell=cell, permanence=INITIAL_PERMANENCE)
            self.synapses.append(new_syn)
    
    def learn(self, strength=1.0) -> None:
        self.adapt(strength)
        self.grow_synapses(strength)
   
    def clear_states(self) -> None:
        self.update_prev_active()
        self.update_prev_learning()
        for syn in self.synapses:
            pass  # Synapses do not have states to clear

class ProximalSynapse(Synapse):
    """Proximal synapse connecting to an input bit."""
    def __init__(self, source_cell: Cell, permanence: float=INITIAL_PERMANENCE) -> None:
        super().__init__(source_cell=source_cell, permanence=permanence)

class Field:
  """A collection of cells."""
  def __init__(self, cells: Iterable[Cell]) -> None:
      self.cells: List[Cell] = list(cells)
      self.prev_active_cells: Set[Cell] = set()

  def __iter__(self):
      return iter(self.cells)
  
  def sample(self, pct: float) -> Set[Cell]:
      """Sample 'pct' percent cells from the field."""
      n = int(len(self.cells) * pct)
      if n > len(self.cells):
          raise ValueError("Cannot sample more cells than are in the field.")
      return set(random.sample(list(self.cells), n))

  def clear_states(self) -> None:
    self.prev_active_cells = self._build_prev_active_cells()
  
  @property
  def active_cells(self) -> Set[Cell]:
      """Return set of previously active cells in the field."""
      return {cell for cell in self.cells if cell.active}

  def _build_prev_active_cells(self) -> Set[Cell]:
      """Return set of previously active cells in the field."""
      return {cell for cell in self.cells if cell.prev_active}
    
  @property
  def predictive_cells(self) -> Set[Cell]:
      """Return set of previously active cells in the field."""
      return {cell for cell in self.cells if cell.predictive}

class Column(Active, Bursting):
    """Column containing cells and proximal synapses for spatial pooling."""
    
    def __init__(
        self,
        input_field: Field = None,
        cells_per_column: int = 1,
    ) -> None:
        super().__init__()
        self.input_field: Field = input_field
        if input_field is not None:
            self.receptive_field: Set[Cell] = self.input_field.sample(RECEPTIVE_FIELD_PCT)
            self.potential_synapses: List[ProximalSynapse] = [ProximalSynapse(source_cell=cell) for cell in self.receptive_field]
            self.connected_synapses: List[ProximalSynapse] = []
            self._update_connected_synapses()
            self.overlap: float = 0.0
        self.cells: List[Cell] = [
            Cell(
                parent_column=self,
            )
            for _ in range(cells_per_column)
        ]
        
    def _update_connected_synapses(self, connected_perm: float = CONNECTED_PERM) -> None:
        """Update the list of connected synapses based on permanence threshold."""
        self.connected_synapses = [s for s in self.potential_synapses 
                                   if s.permanence >= connected_perm]
    
    def compute_overlap(self) -> None:
        """Compute overlap with current binary input vector."""
        self.overlap = sum(s.source_cell.active for s in self.connected_synapses)
                     
        # self.overlap = float(overlap) if overlap >= self.min_overlap else 0.0

    def learn(self) -> None:
      """Learn on proximal synapses based on current input."""
      for syn in self.potential_synapses:
          if syn.source_cell.active:
              syn._adjust_permanence(increase=True)
          else:
              syn._adjust_permanence(increase=False)
      self._update_connected_synapses()

    def get_best_learning_cell(self) -> Cell:
        """Select cell whose segment has most synapses to
           the input cell activations (typically prev active cells);
             fallback to cell with fewest segments.
        """
        best_cell = None
        highest_synapse_count = -1
        
        # Find a cell with segment that has most synapses to cell activations
        for cell in self.cells:
            best_segment = cell.find_best_predictive_segment()
            if best_segment is not None and (synapse_count:= len(best_segment.get_synapses_to_prev_active_cells())) > highest_synapse_count:
                highest_synapse_count = synapse_count
                best_cell = cell
        
        if best_cell is None:
            # Select cell with fewest segments 
            best_cell = min(self.cells, key=lambda c: len(c.segments))
        
        return best_cell

    def clear_states(self) -> None:
        self.update_prev_active()
        self.update_prev_bursting()
        for cell in self.cells:
            cell.clear_states()


class ColumnField(Field):
    def __init__(
        self,
        input_fields: List[Field],
        num_columns: int = 0,
        cells_per_column: int = 1,
        non_spatial: bool = False,
        non_temporal: bool = False,
    ) -> None:
        self.non_spatial = non_spatial
        self.non_temporal = non_temporal
        self.input_field = Field(chain.from_iterable(input_fields))
        if self.non_temporal:
            cells_per_column = 1
        if self.non_spatial:
            num_columns = len(self.input_field.cells)
            self.columns: List[Column] = [
                Column(
                    cells_per_column=cells_per_column,
                )
                for _ in range(num_columns)
            ]
        else:
            self.columns = [
                Column(
                    self.input_field,
                    cells_per_column=cells_per_column,
                )
                for _ in range(num_columns)
            ]
        super().__init__(chain.from_iterable(column.cells for column in self.columns))
        for column in self.columns:
            for cell in column.cells:
              cell.initialize(distal_field=Field(set(self.cells)-{cell}))
        self.active_columns :Column = []
      
    def compute(self, learn=True) -> None:
        self.clear_states()
        
        if self.non_spatial:
            for column, input_cell in zip(self.columns, self.input_field.cells):
                if input_cell.active:
                    self.active_columns.append(column)
                    column.set_active()
        else:
            for column in self.columns:
                column.compute_overlap()
        
            self.activate_columns()

            if learn:
                self.learn_columns()

        if self.non_temporal:
            for column in self.active_columns:
                for cell in column.cells:
                    cell.set_active()
        else:
            self.activate_cells()

            if learn:
                self.learn_cells()

            self.depolarize_cells()
    
    def activate_columns(self) -> None:
        self.activate_top_k_columns(int(len(self.columns) * DESIRED_LOCAL_SPARSITY))

    def learn_columns(self) -> None:
        for column in self.active_columns:
            column.learn()
        
    def activate_top_k_columns(self, k: int) -> None:
        """Activate the top-k columns based on overlap."""
        sorted_columns = sorted(self.columns, key=lambda col: col.overlap, reverse=True)
        for col in sorted_columns[:k]:
            self.active_columns.append(col)
            col.set_active()
    
    def activate_cells(self) -> None:
        for cell in self.predictive_cells:
            if cell.parent_column.active:
                cell.set_active()
        for column in self.active_columns:
            if not any(cell.predictive for cell in column.cells):
                column.set_bursting()
                # Two possible implementations: set all active, or set best matching active
                # column.get_best_matching_cell().set_active()
                for cell in column.cells:
                    cell.set_active()

    def learn_cells(self) -> None:
        for cell in self.predictive_cells:
            # An implementation choice: only learn if cell was active
            # or learn on all predictive cells (strengthen/weaken synapses)
            if cell.active:
                for segment in cell.segments:
                    if segment.prev_active:
                        segment.learn()
        for column in self.active_columns:
            if column.bursting:
                # Implementation choice: only best matching cell learns, or
                column.get_best_learning_cell().get_best_learning_segment().learn() 
                # or all cells learn
                # runs for tooooo long 
                # for cell in column.cells:
                #    cell.get_best_learning_segment().learn()
    
    def depolarize_cells(self) -> None:
        for cell in self.cells:
          cell.update_prev_predictive()
          for segment in cell.segments:
              segment.activate_segment()
    
    @property
    def bursting_columns(self) -> List[Column]:
        """Return list of currently bursting columns."""
        return [column for column in self.columns if column.bursting]
                
    def print_stats(self) -> None:
        """Print statistics about the current stats (with stddev) of the segments  and synapses in the ColumnField."""
        def describe(values: List[float]) -> Tuple[int, float, float, float, float]:
            if not values:
                return 0, 0.0, 0.0, 0.0, 0.0
            count = len(values)
            mean_val = fmean(values)
            std_val = pstdev(values) if count > 1 else 0.0
            return count, mean_val, std_val, min(values), max(values)

        def format_metric(
            label: str,
            stats: Tuple[int, float, float, float, float],
            value_precision: str = ".2f",
            extrema_precision: str = ".0f",
        ) -> str:
            _, mean_val, std_val, min_val, max_val = stats
            mean_str = format(mean_val, value_precision)
            std_str = format(std_val, value_precision)
            min_str = format(min_val, extrema_precision)
            max_str = format(max_val, extrema_precision)
            return (
                f"| {label:<22}| {mean_str:>8} ± {std_str:<8}| {min_str:>8} | {max_str:>8} |"
            )

        segments_per_cell = [len(cell.segments) for cell in self.cells]
        all_segments = [segment for cell in self.cells for segment in cell.segments]
        synapses_per_segment = [len(segment.synapses) for segment in all_segments]
        all_synapses = [syn for segment in all_segments for syn in segment.synapses]
        permanences = [syn.permanence for syn in all_synapses]

        seg_count, seg_mean, seg_std, seg_min, seg_max = describe(segments_per_cell)
        syn_count, syn_mean, syn_std, syn_min, syn_max = describe(synapses_per_segment)
        perm_count, perm_mean, perm_std, perm_min, perm_max = describe(permanences)

        connected_synapses = sum(1 for syn in all_synapses if syn.permanence >= CONNECTED_PERM)
        connected_ratio = (connected_synapses / perm_count) if perm_count else 0.0

        table_lines = [
            "+------------------------+--------------------+----------+----------+",
            "| Metric                 |   Mean ± Std      |      Min |      Max |",
            "+------------------------+--------------------+----------+----------+",
            format_metric("Segments per cell", (seg_count, seg_mean, seg_std, seg_min, seg_max)),
            format_metric("Synapses per segment", (syn_count, syn_mean, syn_std, syn_min, syn_max)),
            format_metric(
                "Permanence",
                (perm_count, perm_mean, perm_std, perm_min, perm_max),
                value_precision=".3f",
                extrema_precision=".3f",
            ),
            "+------------------------+--------------------+----------+----------+",
        ]

        print("ColumnField statistics:")
        print(f"  Columns: {len(self.columns)} | Cells: {len(self.cells)} | Segments: {len(all_segments)} | Synapses: {len(all_synapses)}")
        for line in table_lines:
            print(f"  {line}")
        print(
            f"  Connected synapses (>= {CONNECTED_PERM}): {connected_synapses}"
            f" ({connected_ratio:.1%} of all synapses)"
        )

    def clear_states(self) -> None:
        """Clear active and bursting states for all columns and cells."""
        super().clear_states()
        self.active_columns :List[Column] = []
        for column in self.columns:
            column.clear_states()


class InputField(Field, RandomDistributedScalarEncoder):
    """A Field specialized for input bits."""
    def __init__(self, size, category=False, rdse_params: RDSEParameters=RDSEParameters()) -> None:
        cells = {Cell() for _ in range(size)}
        Field.__init__(self, cells)
        rdse_params.size = size
        rdse_params.category = category
        RandomDistributedScalarEncoder.__init__(self, rdse_params)

    def encode(self, input_value: float) -> List[bool]:
        """Encode the input value into a binary vector."""
        self.clear_states()
        encoded_bits = super().encode(input_value)
        for idx, cell in enumerate(self.cells):
            if encoded_bits[idx]:
                cell.set_active()
        return encoded_bits
    
    def clear_states(self) -> None:
        super().clear_states()
        for cell in self.cells:
            cell.clear_states()

input_field = Field(cells={Cell() for _ in range(10)})

ColumnField(input_fields=[input_field], num_columns=1)  # Dummy instance to avoid linter errors
