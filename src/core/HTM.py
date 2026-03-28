from itertools import chain
import copy
import math
import random
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Set,
    Tuple,
    Optional,
)

from statistics import fmean, pstdev
from encoder_layer.rdse import RDSEParameters


# Constants
CONNECTED_PERM = 0.5  # Permanence threshold for a synapse to be considered connected
DESIRED_LOCAL_SPARSITY = 0.02  # Desired local sparsity for inhibition
INITIAL_PERMANENCE = 0.21  # Initial permanence for new synapses
PERMANENCE_INC = 0.20  # Amount by which synapses are incremented during learning
PERMANENCE_DEC = 0.20  # Amount by which synapses are decremented during learning
PREDICTED_DECREMENT_PCT = 0.1  # Fraction of permanence decrement for predicted but inactive segments
GROWTH_STRENGTH = 0.5  # Fraction of max synapses to grow on a segment during learning
RECEPTIVE_FIELD_PCT = 0.2 # Percentage of distal field sampled by a segment for potential synapses
DUTY_CYCLE_PERIOD = 1000  # Steps used by the duty-cycle moving average
MAX_SYNAPSE_PCT = 0.02  # Max synapses as a percentage of distal field size
ACTIVATION_THRESHOLD_PCT = 0.5  # Activation threshold as a percentage of synapses on segment
LEARNING_THRESHOLD_PCT = 0.25  # Learning threshold as a percentage of synapses on segment

debug = False

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

    def advance_state(self):
        setattr(self, prev_attr, getattr(self, attr))
        setattr(self, attr, False)

    def clear_state(self):
        setattr(self, attr, False)
        setattr(self, prev_attr, False)

    namespace = {
        "__init__": __init__,
        "state_name": attr,
        "prev_state_name": prev_attr,
        f"set_{attr}": set_state,
        "advance_state": advance_state,
        "clear_state": clear_state
    }

    new_class = type(label.capitalize(), (object,), namespace)
    return new_class

Active = make_state_class("active")
Winner = make_state_class("winner")
Predictive = make_state_class("predictive")
Bursting = make_state_class("bursting")
Learning = make_state_class("learning")
Matching = make_state_class("matching")
GoDepolarized = make_state_class("go_depolarized")
NoGoDepolarized = make_state_class("nogo_depolarized")

class Field:
    """A collection of cells."""
    def __init__(self, cells: Iterable['Cell']) -> None:
        self.cells: List['Cell'] = list(cells)

    def __iter__(self):
        return iter(self.cells)

    def sample(self, pct: float) -> Set['Cell']:
        """Sample 'pct' percent cells from the field."""
        n = int(len(self.cells) * pct)
        if n > len(self.cells):
            raise ValueError("Cannot sample more cells than are in the field.")
        return set(random.sample(self.cells, n))
    
    def reset(self) -> None:
        """Reset all cells in the field to initial state."""
        for cell in self.cells:
            cell.clear_state()

    @property
    def active_cells(self) -> Set['Cell']:
        """Return set of currently active cells in the field."""
        return {cell for cell in self.cells if cell.active}

    @property
    def prev_active_cells(self) -> Set['Cell']:
        """Return set of previously active cells in the field."""
        return {cell for cell in self.cells if cell.prev_active}

    @property
    def predictive_cells(self) -> Set['Cell']:
        """Return set of currently predictive cells in the field."""
        return {cell for cell in self.cells if cell.predictive}

    @property
    def prev_predictive_cells(self) -> Set['Cell']:
        """Return set of previously predictive cells in the field."""
        return {cell for cell in self.cells if cell.prev_predictive}

    @property
    def prev_learning_cells(self) -> Set['Cell']:
        """Return set of previously learning cells in the field."""
        return {cell for cell in self.cells if cell.prev_learning}

    @property
    def prev_winner_cells(self) -> Set['Cell']:
        """Return set of previously winning cells in the field."""
        return {cell for cell in self.cells if cell.prev_winner}

# ===== Basic Building Blocks =====

class Synapse:

    def __init__(self, source_cell: 'Cell|None', permanence: float) -> None:
        self.source_cell: 'Cell|None' = source_cell
        self.permanence: float = permanence

    def _adjust_permanence(self, increase: bool, strength: float=1.0) -> None:
        """Adjust synapse permanence by learning rate."""
        if increase:
            self.permanence = min(1.0, self.permanence + PERMANENCE_INC * strength)
        else:
            self.permanence = max(0.0, self.permanence - PERMANENCE_DEC * strength)

    @property
    def active(self) -> bool:
        """Return whether the source cell is currently active."""
        return self.source_cell.active and self.permanence >= CONNECTED_PERM

    @property
    def potentially_active(self) -> bool:
        """Return whether the source cell is currently active, regardless of permanence."""
        return self.source_cell.active and self.permanence > 0.0

    @property
    def prev_active(self) -> bool:
        """Return whether the source cell was previously active."""
        return self.source_cell.prev_active

class ApicalSynapse(Synapse):
    """Distal synapse connecting to a source cell."""

    def __init__(self, source_cell: 'Cell', permanence: float) -> None:
        super().__init__(source_cell, permanence)

class DistalSynapse(Synapse):
    """Distal synapse connecting to a source cell."""

    def __init__(self, source_cell: 'Cell', permanence: float) -> None:
        super().__init__(source_cell, permanence)

class ProximalSynapse(Synapse):
    """Proximal synapse connecting to an input bit."""
    def __init__(self, source_cell: 'Cell', permanence: float=INITIAL_PERMANENCE) -> None:
        super().__init__(source_cell=source_cell, permanence=permanence)

class Segment(Active, Learning, Matching):
    """Segment composed of synapses to cells in a source field."""

    def __init__(
        self,
        parent_cell: 'Cell',
        field: 'Field|None' = None,
        synapses: Optional[List[Synapse]] = None,
        synapse_cls = DistalSynapse
    ) -> None:
        super().__init__()
        self.parent_cell: 'Cell' = parent_cell
        self.field: 'Field' = field if field is not None else parent_cell.distal_field
        self.synapses: List[Synapse] = synapses if synapses is not None else []
        self.sequence_segment: bool = False  # True if learned in a predictive context
        self.max_synapses = int(MAX_SYNAPSE_PCT * len(self.field.cells))
        self.synapse_cls = synapse_cls
        global debug
        if debug:
            print(f"Created Segment with max_synapses={self.max_synapses}")
            debug = False
        self.activation_threshold: float = ACTIVATION_THRESHOLD_PCT
        self.learning_threshold_connected_pct: float = LEARNING_THRESHOLD_PCT

    def is_active(self) -> bool:
        connected_synapses = [syn for syn in self.synapses if syn.active]
        return len(connected_synapses) > self.activation_threshold*len(self.synapses)

    def is_potentially_active(self) -> bool:
        connected_synapses = [syn for syn in self.synapses if syn.potentially_active]
        return len(connected_synapses) > self.learning_threshold_connected_pct*len(self.synapses)

    def potential_prev_active_synapses(self) -> int:
        """Return count of previously active synapses, regardless of permanence."""
        return [syn for syn in self.synapses if syn.source_cell.prev_active]

    def activate_segment(self) -> None:
        if self.is_potentially_active():
            self.set_matching()
            if self.is_active():
                self.set_active()

    def advance_state(self) -> None:
        self.prev_active = self.active
        self.active = False

        self.prev_learning = self.learning
        self.learning = False

        self.prev_matching = self.matching
        self.matching = False

    def clear_state(self) -> None:
        self.active = False
        self.prev_active = False
        self.learning = False
        self.prev_learning = False
        self.matching = False
        self.prev_matching = False

    def adapt(self, strength:float=1.0, active_predicate: Callable[['Synapse'], bool] | None = None) -> None:
        if active_predicate is None:
            active_predicate = lambda syn: syn.source_cell.prev_active
        kept = []
        for syn in self.synapses:
            syn._adjust_permanence(increase=active_predicate(syn), strength=strength)
            if syn.permanence > 0.0:
                kept.append(syn)
        self.synapses = kept

    def grow(self, strength:float=1.0, growth_candidates: Set['Cell'] | None = None) -> None:
        """Grow new synapses to random cells in the source field."""
        growable_synapses = int((self.max_synapses - len(self.synapses))*GROWTH_STRENGTH*strength)
        if growable_synapses > 0:
            if growth_candidates is None:
                growth_candidates = self.field.prev_winner_cells
            potential_cells = list(growth_candidates - {syn.source_cell for syn in self.synapses} - {self.parent_cell})
            random.shuffle(potential_cells)
            cells_to_connect = potential_cells[:growable_synapses]
            for cell in cells_to_connect:
                new_syn = self.synapse_cls(source_cell=cell, permanence=INITIAL_PERMANENCE)
                self.synapses.append(new_syn)

    def weaken(self, strength=1.0) -> None:
        # Weaken synapses to active cells
        # add synpase deletion
        kept = []
        for syn in self.synapses:
            syn._adjust_permanence(increase=False, strength=strength)
            if syn.permanence > 0.0:
                kept.append(syn)
        self.synapses = kept


class ApicalSegment(Segment):
    """Apical segment sampling from a single value field (Go or NoGo).

    Each cell gets one ApicalSegment per value field. The cell combines
    the activation scores from its apical segments into a net depolarization:
    positive net -> go_depolarized, negative net -> nogo_depolarized.

    Learning is TD-error driven via the field's avg_error.
    """

    def __init__(self, parent_cell: 'Cell', field: 'Field', sign: int = 1) -> None:
        super().__init__(parent_cell, field=field, synapse_cls=ApicalSynapse)
        self.sign = sign  # +1 for Go, -1 for NoGo

    def score(self) -> int:
        """Count of active connected synapses."""
        return sum(1 for s in self.synapses if s.active)

    def signed_score(self) -> int:
        """Score weighted by sign (+1 for Go, -1 for NoGo)."""
        return self.sign * self.score()

    def adapt(self, strength: float=1.0) -> None:
        """Adapt synapses using the field's TD error.

        Strengthens synapses to previously active cells when error sign
        matches segment sign (Go strengthens on positive error,
        NoGo strengthens on negative error).
        """
        td_error = self.field.avg_error
        # Go segments learn on positive error, NoGo on negative

        should_strengthen = (self.sign > 0 and td_error > 0) or (self.sign < 0 and td_error < 0)
        if not should_strengthen:
            return

        dec_strength = abs(td_error) * 2.0
        kept = []
        for syn in self.synapses:
            increase = syn.source_cell.prev_active
            s = abs(td_error) if increase else dec_strength
            syn._adjust_permanence(increase=increase, strength=s*strength)
            if syn.permanence > 0.0:
                kept.append(syn)
        self.synapses = kept

class Cell(Active, Winner, Predictive, GoDepolarized, NoGoDepolarized):
    """Single cell within a column or layer.

    Holds distal segments for temporal sequence memory and apical segments
    for reward-modulated voluntary activation. Go and NoGo apical segments
    each sample from their respective value fields; the cell combines their
    scores into a net depolarization number.
    """

    def __init__(
        self,
        parent_column: 'Column|None' = None,
        distal_field: 'Field|None' = None,
        go_field: 'Field|None' = None,
        nogo_field: 'Field|None' = None
    ) -> None:
        super().__init__()
        self.parent_column = parent_column
        self.distal_field = distal_field
        self.go_field = go_field
        self.nogo_field = nogo_field
        self.distal_segments: List[Segment] = []
        self.go_segments: List[ApicalSegment] = []
        self.nogo_segments: List[ApicalSegment] = []
        self.active_duty_cycle: float = 0.0

    @property
    def apical_segments(self) -> List[ApicalSegment]:
        """All apical segments (go + nogo) for iteration."""
        return self.go_segments + self.nogo_segments

    @property
    def segments(self) -> List[Segment]:
        """All distal segments for iteration."""
        return self.distal_segments+self.apical_segments

    def initialize(self, distal_field: 'Field') -> None:
        self.distal_field = distal_field

    def depolarize_apical(self) -> None:
        """Activate apical segments and combine into net go/nogo depolarization."""
        for seg in self.apical_segments:
            seg.activate_segment()

        net = sum(seg.signed_score() for seg in self.apical_segments)
        if net > 0:
            self.set_go_depolarized()
        elif net < 0:
            self.set_nogo_depolarized()
    
    def depolarize_distal(self) -> None:
        """Activate distal segments to set predictive state."""
        for seg in self.distal_segments:
            seg.activate_segment()
            if seg.active:
                self.set_predictive()

    def __repr__(self) -> str:
        return f"Cell(id={id(self)})"

    def advance_state(self) -> None:
        self.prev_active = self.active
        self.active = False

        self.prev_winner = self.winner
        self.winner = False

        self.prev_predictive = self.predictive
        self.predictive = False

        self.prev_go_depolarized = self.go_depolarized
        self.go_depolarized = False

        self.prev_nogo_depolarized = self.nogo_depolarized
        self.nogo_depolarized = False

        for segment in self.distal_segments:
            segment.advance_state()
        for segment in self.apical_segments:
            segment.advance_state()

    def clear_state(self) -> None:
        self.active = False
        self.prev_active = False
        self.winner = False
        self.prev_winner = False
        self.predictive = False
        self.prev_predictive = False
        self.go_depolarized = False
        self.prev_go_depolarized = False
        self.nogo_depolarized = False
        self.prev_nogo_depolarized = False

        for segment in self.segments:
            segment.clear_state()
        for segment in self.segments:
            segment.clear_state()

class Column(Active, Predictive, Bursting):
    """Column containing cells and proximal synapses for spatial pooling."""

    def __init__(
        self,
        input_field: Field|None = None,
        cells_per_column: int = 1,
    ) -> None:
        super().__init__()
        self.input_field: Field|None = input_field
        if input_field is not None:
            self.receptive_field: Set[Cell] = self.input_field.sample(RECEPTIVE_FIELD_PCT)
            self.potential_synapses: List[ProximalSynapse] = [ProximalSynapse(source_cell=cell) for cell in self.receptive_field]
            self.connected_synapses: List[ProximalSynapse] = []
            self._update_connected_synapses()
            self.overlap: float = 0.0
        self.active_duty_cycle: float = 0.0
        self.cells: List[Cell] = [
            Cell(
                parent_column=self,
            )
            for _ in range(cells_per_column)
        ]

    def __repr__(self) -> str:
        return f"Column(id={id(self)})"

    @property
    def distal_segments(self) -> List[Segment]:
        """Return all distal segments on all cells in this column."""
        return list(chain.from_iterable(cell.distal_segments for cell in self.cells))

    @property
    def apical_segments(self) -> List[ApicalSegment]:
        """Return all apical segments on all cells in this column."""
        return list(chain.from_iterable(cell.apical_segments for cell in self.cells))

    def least_used_cell(self, segments_attr: str = 'distal_segments') -> Cell:
        """Return the cell with the fewest segments of the given type."""
        min_segments = min(len(getattr(cell, segments_attr)) for cell in self.cells)
        return random.choice([cell for cell in self.cells if len(getattr(cell, segments_attr)) == min_segments])

    def advance_state(self) -> None:
        self.prev_active = self.active
        self.active = False

        self.prev_bursting = self.bursting
        self.bursting = False

        self.prev_predictive = self.predictive
        self.predictive = False

        for cell in self.cells:
            cell.advance_state()

    def clear_state(self) -> None:
        self.active = False
        self.prev_active = False
        self.bursting = False
        self.prev_bursting = False
        self.predictive = False
        self.prev_predictive = False

        for cell in self.cells:
            cell.clear_state()

    def _update_connected_synapses(self, connected_perm: float = CONNECTED_PERM) -> None:
        """Update the list of connected synapses based on permanence threshold."""
        self.connected_synapses = [s for s in self.potential_synapses
                                   if s.permanence >= connected_perm]

    def compute_overlap(self) -> None:
        """Compute overlap with current binary input vector."""
        self.overlap = sum(s.source_cell.active for s in self.connected_synapses)

    def learn(self) -> None:
      """Learn on proximal synapses based on current input."""
      for syn in self.potential_synapses:
          if syn.source_cell.active:
              syn._adjust_permanence(increase=True)
          else:
              syn._adjust_permanence(increase=False)
      self._update_connected_synapses()

def best_potential_prev_active_segment(segments: List[Segment]) -> Optional[Segment]:
    """Return the previously matching segment with the most previously active potential synapses."""
    best_segment = None
    best_score = -1
    for segment in segments:
        if segment.prev_matching:
            score = len(segment.potential_prev_active_synapses())
            if score > best_score:
                best_score = score
                best_segment = segment
    return best_segment

class ColumnField(Field):
    """A collection of columns."""

    def __init__(
        self,
        input_fields: List[Field],
        num_columns: int = 0,
        cells_per_column: int = 1,
        non_spatial: bool = False,
        non_temporal: bool = False,
        duty_cycle_period: int = DUTY_CYCLE_PERIOD,
        go_field: 'ValueField|None' = None,
        nogo_field: 'ValueField|None' = None,
    ) -> None:
        self.num_columns = num_columns
        self.cells_per_column = cells_per_column
        self.input_fields: List[Field] = list(input_fields)
        self.non_spatial = non_spatial
        self.non_temporal = non_temporal
        self.duty_cycle_period = max(1, duty_cycle_period)
        self._duty_cycle_window = 0
        self._prev_winner_cells: Set[Cell] = set()
        self.go_field = go_field
        self.nogo_field = nogo_field
        self.initialize()

    def initialize(self) -> None:
        self.input_field = Field(chain.from_iterable(self.input_fields))
        if self.non_temporal:
            self.cells_per_column = 1
        if self.non_spatial:
            num_columns = len(self.input_field.cells)
            self.columns: List[Column] = [
                Column(
                    cells_per_column=self.cells_per_column,
                )
                for _ in range(num_columns)
            ]
        else:
            self.columns = [
                Column(
                    self.input_field,
                    cells_per_column=self.cells_per_column,
                )
                for _ in range(self.num_columns)
            ]
        Field.__init__(self, cells=chain.from_iterable(column.cells for column in self.columns))
        for column in self.columns:
            for cell in column.cells:
                cell.initialize(distal_field=self)

        self.clear_states()

    def set_input_fields(self):
        """Set the input fields for this ColumnField."""
        self.input_fields = self.input_fields
        self.initialize()

    def add_input_fields(self, fields: list[Field]) -> None:
        """Add an input field to this ColumnField."""
        self.input_fields.extend(fields)
        additional_cells = Field(chain.from_iterable(field.cells for field in fields))
        self.input_field.cells.extend(additional_cells)
        if self.non_spatial:
            self.columns.extend(Column(cells_per_column=self.cells_per_column)
                                for column in chain.from_iterable(field.cells for field in fields))
            for column in self.columns:
                for cell in column.cells:
                    cell.initialize(distal_field=self)
        else:
            for column in self.columns:
                column.input_field = self.input_field
                column.receptive_field.union(additional_cells.sample(RECEPTIVE_FIELD_PCT))
                column.potential_synapses = [ProximalSynapse(source_cell=cell) for cell in column.receptive_field
                                             if cell not in [syn.source_cell for syn in column.potential_synapses]]
                column._update_connected_synapses()

    def __iter__(self):
        return iter(self.columns)

    @property
    def bursting_columns(self) -> List[Column]:
        """Return list of currently bursting columns."""
        return [column for column in self.columns if column.bursting]

    @property
    def active_columns(self) -> List[Column]:
        """Return list of currently active columns."""
        return [column for column in self.columns if column.active]

    @property
    def prev_winner_cells(self) -> Set[Cell]:
        """Return set of previously winning cells in the field."""
        return self._prev_winner_cells

    def advance_states(self) -> None:
        for cls in ColumnField.__mro__:
            if hasattr(cls, "advance_state") and cls not in (ColumnField, object):
                cls.advance_state(self)
        for column in self.columns:
            column.advance_state()
        self._prev_winner_cells = set(cell for cell in self.cells if cell.prev_winner)

    def clear_states(self) -> None:
        for cls in ColumnField.__mro__:
            if hasattr(cls, "clear_state") and cls not in (ColumnField, object):
                cls.clear_state(self)
        for column in self.columns:
            column.clear_state()
        self._prev_winner_cells = set()

    def compute(self, learn: bool = True) -> None:
        self.advance_states()

        if self.non_spatial:
            for column, input_cell in zip(self.columns, self.input_field.cells):
                if input_cell.active:
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
                self.select_learning_cells(segment_factory=lambda cell: Segment(parent_cell=cell))

            self.depolarize()

            if learn:
                self.learn()

        self.set_prediction()

        self._update_duty_cycles()

    def apical_compute(self, learn: bool = True) -> None:
        self.select_learning_cells(segments_attr='go_segments',
                                   segment_factory=lambda cell: ApicalSegment(parent_cell=cell, field=self.go_field, sign=1))
        self.select_learning_cells(segments_attr='nogo_segments',
                                   segment_factory=lambda cell: ApicalSegment(parent_cell=cell, field=self.nogo_field, sign=-1))
        self.depolarize(segments_attr='apical_segments')
        if learn:
            self.learn(segments_attr='apical_segments')

    def activate_columns(self) -> None:
        self.activate_top_k_columns(int(len(self.columns) * DESIRED_LOCAL_SPARSITY))

    def learn_columns(self) -> None:
        for column in self.active_columns:
            column.learn()

    def activate_top_k_columns(self, k: int) -> None:
        """Activate the top-k columns based on overlap.

        If there are ties at the lowest overlap value in top-k,
        randomly select among the tied columns to meet exactly k.
        """
        sorted_columns = sorted(self.columns, key=lambda col: col.overlap, reverse=True)

        if k >= len(sorted_columns):
            for col in sorted_columns:
                self.active_columns.append(col)
                col.set_active()
            return

        # Find the threshold overlap (the k-th highest value)
        threshold_overlap = sorted_columns[k - 1].overlap

        # Separate columns above threshold from those at threshold
        above_threshold = [col for col in sorted_columns if col.overlap > threshold_overlap]
        at_threshold = [col for col in sorted_columns if col.overlap == threshold_overlap]

        # Activate all columns above threshold
        for col in above_threshold:
            self.active_columns.append(col)
            col.set_active()

        # Randomly select from tied columns to fill remaining spots
        remaining_spots = k - len(above_threshold)
        if remaining_spots > 0 and at_threshold:
            selected = random.sample(at_threshold, remaining_spots)
            for col in selected:
                self.active_columns.append(col)
                col.set_active()

    def activate_cells(self, segments_attr: str = 'distal_segments') -> None:
        for column in self.active_columns:
            if any(cell.prev_predictive for cell in column.cells): # Same as 1) L3
                column.set_predictive()
                for cell in column.cells:
                    for segment in getattr(cell, segments_attr):
                        if segment.prev_active:                        # Same as 1) L11
                            cell.set_active()
                            cell.set_winner()          # Same as 1) L13

            if not any(cell.prev_predictive for cell in column.cells):  # Same as 1) L5
                column.set_bursting()
                for cell in column.cells:
                    cell.set_active()

    def select_learning_cells(self, segments_attr: str = 'distal_segments',
                              segment_factory: Callable[['Cell'], Segment] | None = None) -> None:
        if segment_factory is None:
            segment_factory = Segment
        for column in self.active_columns:
            if column.predictive:
                for cell in column.cells:
                    for segment in getattr(cell, segments_attr):
                        if segment.prev_active:                        # Same as 1) L11
                            segment.set_learning()

            if column.bursting:
                winner_cell = next((cell for cell in column.cells if cell.winner), None)
                if winner_cell is not None: 
                    learning_segment = best_potential_prev_active_segment(getattr(winner_cell, segments_attr))
                    if learning_segment is None:
                        learning_segment = segment_factory(winner_cell)
                        getattr(winner_cell, segments_attr).append(learning_segment)
                else:
                    all_segments = list(chain.from_iterable(
                        getattr(cell, segments_attr) for cell in column.cells
                    ))
                    if any(segment.prev_matching for segment in all_segments):  # Same as 1) L29
                        learning_segment = best_potential_prev_active_segment(all_segments)  # Same as 1) L30
                        winner_cell = learning_segment.parent_cell
                    else:
                        winner_cell = column.least_used_cell(segments_attr)
                        learning_segment = segment_factory(winner_cell)
                        getattr(winner_cell, segments_attr).append(learning_segment)  # Same as 1) L35

                winner_cell.set_winner()              # Same as 2) L37
                learning_segment.set_learning()      # Same as 1) L39

    def depolarize(self, segments_attr: str = 'distal_segments') -> None:
        for column in self.columns:
            for cell in column.cells:
                if segments_attr == 'distal_segments':
                    cell.depolarize_distal()
                elif segments_attr == 'apical_segments':
                    cell.depolarize_apical()

    def learn(self, segments_attr: str = 'distal_segments') -> None:
        for column in self.active_columns:
            if not column.bursting:
                for cell in column.cells:
                    for segment in getattr(cell, segments_attr):
                        if segment.learning:
                            segment.grow()               # Same as 1) L22-24
                            segment.adapt()               # Same as 1) L16-20

        for column in self.bursting_columns:
            for cell in column.cells:
                for segment in getattr(cell, segments_attr):
                    if segment.learning:               # Same as 1) L40-48
                        segment.grow()
                        segment.adapt(strength=1.0)          # Same as 1) L42-44

        for column in self.columns:
            if not column.active:
                for cell in column.cells:
                    for segment in getattr(cell, segments_attr):
                        if segment.matching:
                            segment.weaken(PREDICTED_DECREMENT_PCT)  # Same as 1) L25-27


    def set_prediction(self) -> List[Field]:
        """Propagate predictive state from columns back to input fields."""
        if self.non_spatial:
            for column, input_cell in zip(self.columns, self.input_field):
                if any(cell.predictive for cell in column.cells):
                    input_cell.set_predictive()

            return self.input_fields

    def _update_duty_cycles(self) -> None:
        self._duty_cycle_window = min(self.duty_cycle_period, self._duty_cycle_window + 1)
        alpha = 1.0 / self._duty_cycle_window
        for column in self.columns:
            column.active_duty_cycle += alpha * ((1.0 if column.active else 0.0) - column.active_duty_cycle)
        for cell in self.cells:
            cell.active_duty_cycle += alpha * ((1.0 if cell.active else 0.0) - cell.active_duty_cycle)

    def print_stats(self) -> None:
        """Print statistics about segments and synapses in the ColumnField."""
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

        segments_per_cell = [len(cell.distal_segments) for cell in self.cells]
        all_segments = [segment for cell in self.cells for segment in cell.distal_segments]
        synapses_per_segment = [len(segment.synapses) for segment in all_segments]
        all_synapses = [syn for segment in all_segments for syn in segment.synapses]
        permanences = [syn.permanence for syn in all_synapses]
        column_duty_cycles = [column.active_duty_cycle for column in self.columns]
        cell_duty_cycles = [cell.active_duty_cycle for cell in self.cells]

        seg_count, seg_mean, seg_std, seg_min, seg_max = describe(segments_per_cell)
        syn_count, syn_mean, syn_std, syn_min, syn_max = describe(synapses_per_segment)
        perm_count, perm_mean, perm_std, perm_min, perm_max = describe(permanences)
        col_duty_stats = describe(column_duty_cycles)
        cell_duty_stats = describe(cell_duty_cycles)

        connected_synapses = sum(1 for syn in all_synapses if syn.permanence >= CONNECTED_PERM)
        connected_ratio = (connected_synapses / perm_count) if perm_count else 0.0
        active_columns = sum(1 for duty in column_duty_cycles if duty > 0.0)
        active_cells = sum(1 for duty in cell_duty_cycles if duty > 0.0)
        column_share = (active_columns / len(self.columns)) if self.columns else 0.0
        cell_share = (active_cells / len(self.cells)) if self.cells else 0.0

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
            format_metric(
                "Column duty cycle",
                col_duty_stats,
                value_precision=".3f",
                extrema_precision=".3f",
            ),
            format_metric(
                "Cell duty cycle",
                cell_duty_stats,
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
        print(
            f"  Columns with duty > 0: {active_columns}/{len(self.columns)} ({column_share:.1%})"
        )
        print(
            f"  Cells with duty > 0: {active_cells}/{len(self.cells)} ({cell_share:.1%})"
        )

class InputField(Field):
    """A Field specialized for input bits."""

    def __init__(self, encoder_params: Any | None = None, size: int | None = None) -> None:
        params = copy.deepcopy(encoder_params) if encoder_params is not None else RDSEParameters()
        if size is not None and hasattr(params, "size"):
            params.size = size
        self.encoder = params.encoder_class(params)
        cells = {Cell() for _ in range(self.encoder.size)}
        Field.__init__(self, cells)

    def encode(self, input_value: Any) -> List[int]:
        """Encode the input value into a binary vector."""
        self.advance_states()
        encoded_bits = self.encoder.encode(input_value)
        for idx, cell in enumerate(self.cells):
            if encoded_bits[idx]:
                cell.set_active()
        return encoded_bits

    def decode(self, state :str='active', encoded: Field=None, candidates: Iterable[float] | None = None) -> Tuple[float | None]:
        """Convert active cells back to input value using RDSE decoding."""
        if state not in ('active', 'predictive'):
            raise ValueError(f"Invalid state '{state}'; must be 'active' or 'predictive'")
        if encoded is None:
            encoded = self.cells
        self.bit_vector = [getattr(cell, state)  for cell in encoded]
        return self.encoder.decode(self.bit_vector, candidates)

    def advance_states(self) -> None:
        for cell in self.cells:
            cell.advance_state()

    def clear_states(self) -> None:
        for cell in self.cells:
            cell.clear_state()

class OutputField(InputField):
    """InputField-like output layer with learnable connections to a source field.

    - Behaves like InputField for encode/decode/state handling.
    - Each output cell has learnable synapses to cells in `input_field`.
    - Random activation is modulated by connected Go and NoGo depolarizations.
    """

    def __init__(
        self,
        input_field: Field,
        encoder_params: Any | None = None,
        size: int | None = None,
        base_activation_probability: float = 0.1,
        go_gain: float = 1.0,
        nogo_gain: float = 1.0,
        connected_perm: float = CONNECTED_PERM,
        decode_confidence_threshold: float = 0.5,
        random_action_picker: Callable[[list[Any]], Any] | None = None,
    ) -> None:
        if encoder_params is None:
            encoder_size = size if size is not None else max(1, len(input_field.cells))
            active_bits = max(1, min(16, encoder_size))
            encoder_params = RDSEParameters(
                size=encoder_size,
                active_bits=active_bits,
                sparsity=0.0,
                radius=0.0,
                resolution=1.0,
                category=False,
                seed=1,
            )
        super().__init__(encoder_params=encoder_params, size=size)
        self.input_field = input_field
        self.base_activation_probability = base_activation_probability
        self.go_gain = go_gain
        self.nogo_gain = nogo_gain
        self.connected_perm = connected_perm
        self.decode_confidence_threshold = max(0.0, min(1.0, decode_confidence_threshold))
        self.random_action_picker = random_action_picker
        for cell in self.cells:
            cell.distal_field = self.input_field
            segment = Segment(parent_cell=cell, field=self.input_field)
            segment.max_synapses = len(self.input_field.cells)
            cell.distal_segments = [segment]

    def _connected_go_count(self, segment: Segment) -> int:
        return sum(
            1
            for synapse in segment.synapses
            if synapse.permanence >= self.connected_perm and synapse.source_cell.go_depolarized
        )

    def _connected_nogo_count(self, segment: Segment) -> int:
        return sum(
            1
            for synapse in segment.synapses
            if synapse.permanence >= self.connected_perm and synapse.source_cell.nogo_depolarized
        )

    def _activate_cells_from_action(self, action: Any) -> None:
        encoded_bits = self.encoder.encode(action)
        # TODO Could optimize by directly activating cells without encoding the full vector first
        for idx, cell in enumerate(self.cells):
            cell.active = False
            if encoded_bits[idx]:
                cell.set_active()

    def _pick_random_action(self) -> Any | None:
        candidates = self._encoder_action_candidates()
        if not candidates:
            return None
        if self.random_action_picker is not None:
            return self.random_action_picker(candidates)
        return random.choice(candidates)

    def _encoder_action_candidates(self) -> list[Any]:
        return list(self.encoder.encoding_cache.keys())

    def compute(self, learn: bool = True) -> None:
        self.advance_states()

        action = self.decode_from_probabilities(probabilities=self.activation_probabilities())
        if action['confidence'] >= self.decode_confidence_threshold:
            self._activate_cells_from_action(action['value'])
            print('NON-RANDOM ACTION SELECTED')
        else:
            random_action = self._pick_random_action()
            if random_action is not None:
                self._activate_cells_from_action(random_action)

        if learn:
            self.learn()

    def learn(self) -> None:
        growth_targets = set(
            cell for cell in self.input_field.cells
            if cell.go_depolarized or cell.nogo_depolarized
        )
        for cell in self.cells:
            if cell.prev_active:
                segment = cell.distal_segments[0]
                segment.adapt(active_predicate=lambda syn: syn.source_cell.go_depolarized or syn.source_cell.nogo_depolarized)
                if growth_targets:
                    segment.grow(growth_candidates=growth_targets)


    def decode(self, state: str = "active", encoded: Field = None, candidates: Iterable[Any] | None = None):
        decoded_value = None
        confidence = None
        try:
            decoded_value, confidence = InputField.decode(
                self,
                state=state,
                encoded=encoded,
                candidates=candidates,
            )
        except ValueError:
            pass
        return {
            "value": decoded_value,
            "confidence": confidence,
        }

    def activation_probabilities(self) -> List[float]:
        """Compute the activation probability of each output cell from Go/NoGo modulation.

        Uses the same formula as compute() but returns the continuous
        probabilities instead of stochastically sampling them.
        """
        source_cell_count = max(1, len(self.input_field.cells))
        probabilities: List[float] = []
        for cell in self.cells:
            segment = cell.distal_segments[0]
            connected_go = self._connected_go_count(segment)
            connected_nogo = self._connected_nogo_count(segment)
            go_mod = self.go_gain * (connected_go / source_cell_count)
            nogo_mod = self.nogo_gain * (connected_nogo / source_cell_count)
            prob = max(0.0, min(1.0, self.base_activation_probability + go_mod - nogo_mod))
            probabilities.append(prob)
        return probabilities

    def decode_from_probabilities(
        self,
        probabilities: List[float] | None = None,
        candidates: Iterable[Any] | None = None,
    ) -> dict:
        """Decode using continuous activation probabilities instead of binary cell states.

        For each candidate encoding, computes a weighted overlap by summing the
        activation probabilities at positions where the encoding has a 1-bit.
        Returns the candidate with the highest expected overlap.

        Args:
            probabilities: Per-cell activation probabilities. If None, computed
                from the current Go/NoGo modulation state.
            candidates: Values to consider. Defaults to the encoder's cache.
        """
        if probabilities is None:
            probabilities = self.activation_probabilities()

        if len(probabilities) != len(self.cells):
            raise ValueError(
                f"probabilities length ({len(probabilities)}) must match "
                f"cell count ({len(self.cells)})"
            )

        search_values = (
            list(candidates) if candidates is not None
            else list(getattr(self.encoder, "_encoding_cache", {}).keys())
        )
        if not search_values:
            return {"value": None, "confidence": 0.0, "probabilities": probabilities}

        n = len(probabilities)
        mean_p = sum(probabilities) / n

        best_value: Any | None = None
        best_score = -float('inf')
        best_encoding = None

        for candidate in search_values:
            encoding = self.encoder._encoding_cache.get(candidate)
            if encoding is None:
                encoding = self.encoder.register_encoding(candidate)
            # Score using mean-centered probabilities correlated with
            # mean-centered encoding. This isolates the signal from the
            # uniform baseline when probabilities are tightly distributed.
            mean_e = sum(encoding) / n
            score = sum(
                (p - mean_p) * (b - mean_e)
                for p, b in zip(probabilities, encoding)
            )
            if score > best_score:
                best_score = score
                best_value = candidate
                best_encoding = encoding

        # Confidence: Pearson correlation between probabilities and the best
        # candidate's encoding. Measures how well the pattern of deviations
        # in probabilities aligns with the candidate's activation pattern,
        # independent of the baseline probability level.
        mean_e = sum(best_encoding) / n
        var_p = sum((p - mean_p) ** 2 for p in probabilities)
        var_e = sum((b - mean_e) ** 2 for b in best_encoding)
        denom = math.sqrt(var_p * var_e)
        confidence = best_score / denom if denom > 0 else 0.0

        return {
            "value": best_value,
            "confidence": confidence,
        }
