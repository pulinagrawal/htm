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

# Cell class represents a single cell in a column
class Cell:
    """Single cell within a column.

    Holds a (possibly empty) list of distal segments used for Temporal Memory.
    """

    segments: List['Segment']  # forward reference
    active:bool

    def __init__(self) -> None:
        self.segments = []
        self.active = False

    def __repr__(self) -> str:
        return f"Cell(id={id(self)})"


class DistalSynapse:
    """Distal synapse referencing a source cell (Temporal Memory)."""

    source_cell: Cell
    permanence: float

    def __init__(self, source_cell: Cell, permanence: float) -> None:
        self.source_cell = source_cell
        self.permanence = permanence

# Synapse class represents a synapse which connects a segment to a source input
class Segment:
    """Distal segment composed of synapses to previously active cells."""

    synapses: List[DistalSynapse]
    sequence_segment: bool

    def __init__(self, synapses: Optional[List[DistalSynapse]] = None) -> None:
        self.synapses = synapses if synapses is not None else []
        self.sequence_segment = False  # True if learned in a predictive context

    def active_synapses(self, active_cells: Set[Cell]) -> List[DistalSynapse]:
        """Return connected synapses whose source cell is active."""
        return [syn for syn in self.synapses if syn.source_cell in active_cells and syn.permanence > CONNECTED_PERM]

    def matching_synapses(self, prev_active_cells: Set[Cell]) -> List[DistalSynapse]:
        """Return synapses whose source cell was previously active (ignores permanence threshold)."""
        return [syn for syn in self.synapses if syn.source_cell in prev_active_cells]

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
