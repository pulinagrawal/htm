import numpy as np

# Constants (Spatial Pooler)
CONNECTED_PERM = 0.5  # Permanence threshold for an input synapse to be considered connected
MIN_OVERLAP = 3       # Minimum overlap to be considered during inhibition
PERMANENCE_INC = 0.01
PERMANENCE_DEC = 0.01
DESIRED_LOCAL_ACTIVITY = 10

# Constants (Temporal Memory)
SEGMENT_ACTIVATION_THRESHOLD = 3   # Active connected distal synapses required for a segment to be active (prediction)
SEGMENT_LEARNING_THRESHOLD = 3     # Subset used to select best matching segment
INITIAL_DISTAL_PERM = 0.21         # Initial permanence for new distal synapses
NEW_SYNAPSE_MAX = 6                # New synapses to add when reinforcing a learning segment

input_space_size = 100


# Cell class represents a single cell in a column
class Cell:
    def __init__(self):
        # Distal segments for temporal memory
        self.segments = []  # Initially empty; segments created during bursting learning

    def __repr__(self):  # Helpful for debugging / tests
        return f"Cell(id={id(self)})"


class DistalSynapse:
    """Distal synapse referencing a source cell (temporal memory)."""
    def __init__(self, source_cell, permanence):
        self.source_cell = source_cell
        self.permanence = permanence


class Segment:
    def __init__(self, synapses=None):
        self.synapses = synapses if synapses is not None else []  # List[DistalSynapse]
        self.sequence_segment = False  # True if learned in a predictive context

    def active_synapses(self, active_cells):
        return [syn for syn in self.synapses if syn.source_cell in active_cells and syn.permanence > CONNECTED_PERM]

    def matching_synapses(self, prev_active_cells):  # includes synapses even if below connection threshold
        return [syn for syn in self.synapses if syn.source_cell in prev_active_cells]

# Synapse class represents a synapse which connects a segment to a source input
class Synapse:
    """Proximal synapse (input space) used by Spatial Pooler only."""
    def __init__(self, source_input, permanence):
        self.source_input = source_input
        self.permanence = permanence

# Column class represents a column in the HTM Region
class Column:
    def __init__(self, potential_synapses, position):
        self.position = position
        self.potential_synapses = potential_synapses  # List of potential synapses for the column
        self.boost = 1  # Boost factor for column activity
        self.active_duty_cycle = 0
        self.overlap_duty_cycle = 0
        self.min_duty_cycle = 0.01  # Minimum desired firing rate for a column
        # Connected synapses are those whose permanence value is above the threshold
        self.connected_synapses = [s for s in potential_synapses if s.permanence > CONNECTED_PERM]
        self.overlap = 0  # Overlap of column with the current input

    # Computes the overlap of the column with the current input and applies boosting
    def compute_overlap(self, input_vector):
        overlap = sum(1 for s in self.connected_synapses if input_vector[s.source_input])
        self.overlap = overlap * self.boost if overlap >= MIN_OVERLAP else 0
        print(f"Column at position {self.position} has overlap: {self.overlap}")
        
print("Starting the Temporal Pooler process...")        

class TemporalPooler:
    def __init__(self, input_space_size, column_count, cells_per_column, initial_synapses_per_column):
        # Spatial Pooler region (proximal synapses)
        self.input_space_size = input_space_size
        self.columns = self.initialize_region(input_space_size, column_count, initial_synapses_per_column)
        self.cells_per_column = cells_per_column
        for c in self.columns:
            c.cells = [Cell() for _ in range(cells_per_column)]

        # Temporal Memory state (time-indexed)
        self.active_cells = {}        # t -> set of active cells
        self.winner_cells = {}        # t -> set of winner cells (cells chosen for learning in each active column)
        self.predictive_cells = {}    # t -> set of predictive cells (predicting activation at t+1)
        self.learning_segments = {}   # t -> set of segments selected for positive learning
        self.negative_segments = {}   # t -> set of segments to punish (predicted but column inactive)
        # Multi-field metadata
        self.field_ranges = {}  # name -> (start_index, end_index) half-open
        self.field_order = []   # preserve order for reconstruction
        self.column_field_map = {}  # column -> dominant field name

    # Initializes the columns in the region
    def initialize_region(self, input_space_size, column_count, initial_synapses_per_column):
        columns = []
        grid_size = int(column_count ** 0.5)  # Assuming a square grid for simplicit
        for i in range(column_count):
            x = i % grid_size
            y = i // grid_size
            position = (x, y)
            potential_synapses = [Synapse(np.random.randint(input_space_size), np.random.uniform(0.4, 0.6)) for _ in range(initial_synapses_per_column)]
            columns.append(Column(potential_synapses, position))  # Pass position when creating a Column
        print(f"Initialized {len(columns)} columns with positions and potential synapses.")
        return columns
        

    # Computes the active columns after applying inhibition
    def compute_active_columns(self, input_vector, inhibition_radius):
        """Compute active columns from a single input vector OR multiple coding fields.
        Accepts either:
          - 1D numpy array / list of ints (binary)
          - list/tuple of such arrays representing distinct coding fields which will
            be concatenated internally into a single input space.
          - dict mapping field name -> array, concatenated in insertion order; field
            name / ranges stored for later filtering of predictive columns.
        NOTE: To use multiple fields reliably, initialize the TemporalPooler with
        input_space_size equal to the concatenated length of all fields that will
        be supplied, OR only supply fields whose total length equals the original
        size. A ValueError is raised otherwise.
        """
        combined = self._combine_input_fields(input_vector)
        if combined.shape[0] != self.input_space_size:
            raise ValueError(
                f"Combined input length {combined.shape[0]} != configured input_space_size {self.input_space_size}."
            )
        # If dictionary-based input used, (re)assign column field ownership
        if self.field_ranges and any(len(rg) != 2 for rg in self.field_ranges.values()):
            # Defensive: ensure structure is consistent (should not happen), reset
            self.field_ranges = {}
            self.field_order = []
        if self.field_ranges:
            # field metadata already exists; if new dict with different structure passed later, user should reset state.
            pass
        # Build / update column->field mapping if we have field metadata and haven't assigned yet
        if self.field_ranges and not self.column_field_map:
            self._assign_column_fields()
        for c in self.columns:
            c.compute_overlap(combined)
        active_columns = self.inhibition(self.columns, inhibition_radius)
        print(f"Computed active columns. Total active columns: {len(active_columns)}")
        return active_columns

    def _combine_input_fields(self, input_vector):
        """Internal helper: if input_vector is a list/tuple of fields, concatenate.
        Also allows a dict of name -> field which records field ranges.
        Ensures binary int np array output."""
        # Dict branch (ordered in insertion order)
        if isinstance(input_vector, dict):
            start = 0
            arrays = []
            # Reset metadata for this composition
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
            # Column-field map will be recomputed
            self.column_field_map = {}
            return combined
        if isinstance(input_vector, (list, tuple)):
            arrays = [np.asarray(v, dtype=int) for v in input_vector]
            return np.concatenate(arrays) if arrays else np.array([], dtype=int)
        return np.asarray(input_vector, dtype=int)

    def _assign_column_fields(self):
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
    
    # --------------------- Temporal Memory Core ---------------------
    def compute_active_state(self, active_columns, t):
        """Compute active & winner cells at time t using predictive cells from t-1.
        If a column was predicted (one or more predictive cells at t-1) only those predictive cells become active.
        Otherwise the column bursts (all cells active) and we pick a learning cell (winner cell)."""
        prev_predictive = self.predictive_cells.get(t-1, set())
        active_cells_t = set()
        winner_cells_t = set()
        learning_segments_t = set()

        for column in active_columns:
            # Predictive cells from previous time in this column
            predictive_cells_prev = [cell for cell in column.cells if cell in prev_predictive]
            if predictive_cells_prev:
                # Column correctly predicted
                for cell in predictive_cells_prev:
                    active_cells_t.add(cell)
                    winner_cells_t.add(cell)
                    # Mark segments that were active at t-1 for learning
                    for seg in self.active_segments_of(cell, t-1):
                        learning_segments_t.add(seg)
            else:
                # Bursting: all cells active
                for cell in column.cells:
                    active_cells_t.add(cell)
                # Choose learning (winner) cell
                best_cell, best_segment = self.best_matching_cell(column, t-1)
                if best_segment is None:
                    # Create a new segment (ensure best_cell not None)
                    if best_cell is None:
                        # Fallback: choose first cell
                        best_cell = column.cells[0]
                    best_segment = Segment()
                    best_cell.segments.append(best_segment)
                winner_cells_t.add(best_cell)
                learning_segments_t.add(best_segment)

        self.active_cells[t] = active_cells_t
        self.winner_cells[t] = winner_cells_t
        self.learning_segments[t] = learning_segments_t
        print(f"Active state computed for time step {t}: {len(active_cells_t)} cells active.")

    def compute_predictive_state(self, t):
        """Compute predictive cells for next time based on segments active at time t.
        A segment is active if it has enough active connected synapses whose source cells are active at t."""
        active_cells_t = self.active_cells.get(t, set())
        predictive_cells_t = set()
        for column in self.columns:
            for cell in column.cells:
                for seg in cell.segments:
                    if len(seg.active_synapses(active_cells_t)) >= SEGMENT_ACTIVATION_THRESHOLD:
                        predictive_cells_t.add(cell)
                        break
        self.predictive_cells[t] = predictive_cells_t
        print(f"Predictive state computed for time step {t}: {len(predictive_cells_t)} cells predictive.")

    def get_predictive_columns(self, t=None, field_name=None):
        """Return predicted columns.

        Parameters:
          t: int or None
             - None (default): use latest computed predictive time step.
             - -1: previous step relative to latest.
             - any other int: use that exact time index.
          field_name: optional string. If provided and dictionary-based input was
             previously used (field_ranges set), only columns whose dominant field
             matches are returned. If field metadata absent, raises ValueError.

        Returns: set of Column objects.
        """
        if not self.predictive_cells:
            return set()
        max_t = max(self.predictive_cells.keys())
        if t is None:
            query_t = max_t
        elif t == -1:
            query_t = max_t - 1
        else:
            query_t = t
        if query_t < 0:
            return set()
        pred_cells = self.predictive_cells.get(query_t, set())
        cols = {col for col in self.columns if any(cell in pred_cells for cell in col.cells)}
        if field_name is not None:
            if not self.field_ranges:
                raise ValueError("Field-specific prediction requested but no field metadata available (no dict input used).")
            if not self.column_field_map:
                self._assign_column_fields()
            cols = {c for c in cols if self.column_field_map.get(c) == field_name}
        return cols

    def reset_state(self):
        """Reset transient temporal memory state (cells' learned segments remain)."""
        self.active_cells = {}
        self.winner_cells = {}
        self.predictive_cells = {}
        self.learning_segments = {}
        self.negative_segments = {}

    def learn(self, t):
        """Apply learning updates after computing active and predictive states at time t.
        - Reinforce learning segments for winner cells (positive).
        - Punish segments that predicted (at t-1) but whose columns did not become active (negative)."""
        prev_predictive = self.predictive_cells.get(t-1, set())
        active_columns = {c for c in self.columns if any(cell in self.active_cells.get(t, set()) for cell in c.cells)}
        # Negative segments: segments that were active at t-1 (causing prediction) but column not active at t
        negative_segments = set()
        for column in self.columns:
            if column not in active_columns:
                for cell in column.cells:
                    if cell in prev_predictive:
                        for seg in self.active_segments_of(cell, t-1):
                            negative_segments.add(seg)
        self.negative_segments[t] = negative_segments

        # Positive reinforcement
        for seg in self.learning_segments.get(t, set()):
            self.reinforce_segment(seg, t)
        # Negative reinforcement
        for seg in negative_segments:
            self.punish_segment(seg, t)
        print(f"Learning applied at time {t}: +{len(self.learning_segments.get(t, set()))} / -{len(negative_segments)} segments.")

    # --------------------- Helper Methods (TM) ---------------------
    def best_matching_cell(self, column, prev_t):
        prev_active_cells = self.active_cells.get(prev_t, set())
        best_cell = None
        best_segment = None
        best_match = -1
        for cell in column.cells:
            if not cell.segments:
                # Prefer an unused cell immediately if no better match found yet
                if best_match == -1:
                    best_cell = cell
                    best_segment = None
                    best_match = 0
                continue
            for seg in cell.segments:
                match_count = len(seg.matching_synapses(prev_active_cells))
                if match_count > best_match and match_count >= 0:
                    best_match = match_count
                    best_cell = cell
                    best_segment = seg
        return best_cell, best_segment

    def active_segments_of(self, cell, t):
        prev_active_cells = self.active_cells.get(t, set())
        active_list = []
        for seg in cell.segments:
            if len(seg.active_synapses(prev_active_cells)) >= SEGMENT_ACTIVATION_THRESHOLD:
                active_list.append(seg)
        return active_list

    def reinforce_segment(self, segment, t):
        prev_active_cells = self.active_cells.get(t-1, set())
        # Strengthen existing active synapses
        for syn in segment.synapses:
            if syn.source_cell in prev_active_cells:
                syn.permanence = min(1.0, syn.permanence + PERMANENCE_INC)
            else:
                syn.permanence = max(0.0, syn.permanence - PERMANENCE_DEC)
        # Grow new synapses (sample from prev active cells not already connected)
        existing_sources = {syn.source_cell for syn in segment.synapses}
        candidates = [c for c in prev_active_cells if c not in existing_sources]
        np.random.shuffle(candidates)
        for cell_src in candidates[:NEW_SYNAPSE_MAX]:
            segment.synapses.append(DistalSynapse(cell_src, INITIAL_DISTAL_PERM))
        segment.sequence_segment = True

    def punish_segment(self, segment, t):
        for syn in segment.synapses:
            syn.permanence = max(0.0, syn.permanence - PERMANENCE_DEC)



    # Applies inhibition to determine which columns will become active
    def inhibition(self, columns, inhibition_radius):
        active_columns = []
        for c in columns:
        # Find neighbors of column c
            neighbors = [c2 for c2 in columns if c != c2 and self.euclidean_distance(c.position, c2.position) <= inhibition_radius]
            min_local_activity = self.kth_score(neighbors, DESIRED_LOCAL_ACTIVITY)
            if c.overlap > 0 and c.overlap >= min_local_activity:
                active_columns.append(c)
        print(f"After inhibition, active columns: {[c.position for c in active_columns]}")
        return active_columns

    def euclidean_distance(self, pos1, pos2):
    # Calculate the Euclidean distance between two points (pos1 and pos2)
        return np.linalg.norm(np.array(pos1) - np.array(pos2))


    # Returns the k-th highest overlap value from a list of columns
    def kth_score(self, neighbors, k):
        if not neighbors:
            return 0
        ordered = sorted(neighbors, key=lambda x: x.overlap, reverse=True)
        if k <= 0:
            return 0
        if k > len(ordered):
            # If fewer neighbors than desired local activity, use lowest neighbor overlap
            return ordered[-1].overlap if ordered else 0
        return ordered[k-1].overlap


    # Removed legacy helper methods tied to old temporal representation.


    
    def learning_phase(self, active_columns, input_vector):                                   # Learning for Spatial Pooler (unchanged)
        for c in active_columns:
            for s in c.potential_synapses:
                if input_vector[s.source_input]:
                    s.permanence = min(1.0, s.permanence + PERMANENCE_INC)
                else:
                    s.permanence = max(0.0, s.permanence - PERMANENCE_DEC)
            c.connected_synapses = [s for s in c.potential_synapses if s.permanence > CONNECTED_PERM]
        print(f"Learning phase updated the synapses for {len(active_columns)} active columns.")

        # Could update inhibition radius if implementing adaptive inhibition.
        _ = self.average_receptive_field_size(self.columns)

    def average_receptive_field_size(self, columns):
        total_receptive_field_size = 0
        count = 0
        for c in columns:
            connected_positions = [s.source_input for s in c.connected_synapses]
            if connected_positions:
                receptive_field_size = max(connected_positions) - min(connected_positions)
                total_receptive_field_size += receptive_field_size
                count += 1
        return total_receptive_field_size / count if count > 0 else 0
    



if __name__ == "__main__":
    # Example usage of combined Spatial Pooler + revised Temporal Memory
    input_space_size = 100
    column_count = 256  # reduced for demo runtime
    cells_per_column = 8
    initial_synapses_per_column = 20
    steps = 5
    inhibition_radius = 10

    tp = TemporalPooler(input_space_size, column_count, cells_per_column, initial_synapses_per_column)

    for t in range(steps):
        input_vector = np.random.randint(2, size=input_space_size)
        active_columns = tp.compute_active_columns(input_vector, inhibition_radius)
        tp.compute_active_state(active_columns, t)
        tp.compute_predictive_state(t)
        tp.learn(t)
        if t == 0:
            # Spatial pooler learning only occasionally (for demo call once)
            tp.learning_phase(active_columns, input_vector)
    print("Temporal Memory demo completed.")