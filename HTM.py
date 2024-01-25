import numpy as np

# Constants
CONNECTED_PERM = 0.5  # Defines the permanence threshold for a synapse to be considered connected
MIN_OVERLAP = 3       # Minimum number of synapses that must be active for a column to be considered during the inhibition step
PERMANENCE_INC = 0.01  # Amount by which synapses are incremented during learning
PERMANENCE_DEC = 0.01  # Amount by which synapses are decremented during learning
DESIRED_LOCAL_ACTIVITY = 10  # The number of columns that will be winners after the inhibition step

# Cell class represents a single cell in a column
class Cell:
    def __init__(self):
        self.segments = []  # List of segments for the cell
# Segment class represents a segment that can form synapses with other cells
class Segment:
    def __init__(self):
        self.synapses = []  # List of synapses in the segment
        self.sequence_segment = False  # Flag indicating if the segment predicts sequence

# Synapse class represents a synapse which connects a segment to a source input
class Synapse:
    def __init__(self, source_input, permanence):
        self.source_input = source_input  # Index of the input bit
        self.permanence = permanence  # Permanence value of the synapse

# Column class represents a column in the HTM Region
class Column:
    def __init__(self, potential_synapses):
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

class TemporalPooler:
    def __init__(self, input_space_size, column_count, cells_per_column, initial_synapses_per_column):
        # Initialize columns with potential synapses
        self.columns = self.initialize_region(input_space_size, column_count, initial_synapses_per_column)
        self.cells_per_column = cells_per_column
        # States for active, predictive and learning
        self.active_state = {}
        self.predictive_state = {}
        self.learn_state = {}
        self.segment_update_list = {}
        self.THRESHOLD = 5  # Threshold for a segment to be considered active

        # Initialize cells for each column
        for c in self.columns:
            c.cells = [Cell() for _ in range(cells_per_column)]

    # Initializes the columns in the region
    def initialize_region(self, input_space_size, column_count, initial_synapses_per_column):
        columns = []
        for _ in range(column_count):
            potential_synapses = [Synapse(np.random.randint(input_space_size), np.random.uniform(0.4, 0.6)) for _ in range(initial_synapses_per_column)]
            columns.append(Column(potential_synapses))
        return columns

    # Computes the active columns after applying inhibition
    def compute_active_columns(self, input_vector, inhibition_radius):
        for c in self.columns:
            c.compute_overlap(input_vector)
        return self.inhibition(self.columns, inhibition_radius)

    # Applies inhibition to determine which columns will become active
    def inhibition(self, columns, inhibition_radius):
        active_columns = []
        for c in columns:
            neighbors = [c2 for c2 in columns if c != c2 and np.linalg.norm(c2.position - c.position) <= inhibition_radius]
            min_local_activity = self.kth_score(neighbors, DESIRED_LOCAL_ACTIVITY)
            if c.overlap > 0 and c.overlap >= min_local_activity:
                active_columns.append(c)
        return active_columns

    # Returns the k-th highest overlap value from a list of columns
    def kth_score(self, neighbors, k):
        return sorted(neighbors, key=lambda x: x.overlap, reverse=True)[k-1].overlap


    def get_active_segment(self, column_index, cell_index, t, state):                         # Returns the segment of a cell that was active in the last time step

        cell = self.columns[column_index].cells[cell_index]
        for segment in cell.segments:
            if self.segment_active(segment, t-1, state):
                return segment
        return None

    def segment_active(self, segment, t, state):                                              # Determines if a segment is active based on the activity of its synapses at time step t
        active_synapses = self.get_active_synapses(segment, t, state)
        if len(active_synapses) >= self.THRESHOLD:
            return True
        return False

    def get_best_matching_segment(self, column_index, cell_index, t):                         # Finds the segment with the largest number of active synapses. This method is part of the learning phase.
        best_segment = None
        max_overlap = 0
        cell = self.columns[column_index].cells[cell_index]
        for segment in cell.segments:
            overlap = self.get_active_synapses(segment, t, self.active_state)
            if len(overlap) > max_overlap:
                best_segment = segment
                max_overlap = len(overlap)
        return best_segment

    def get_active_synapses(self, segment, t, state):                                         # Returns the list of active synapses in a segment at time step t
        active_synapses = []
        for synapse in segment.synapses:
            if state.get((synapse.source_input, t), False):
                active_synapses.append(synapse)
        return active_synapses

    def adapt_segments(self, segment_list, positive_reinforcement):                           # Update the permanence value of the synapses in a segment based on the reinforcement signal
        for segment in segment_list:
            for synapse in segment.synapses:
                if positive_reinforcement:
                    synapse.permanence = min(1.0, synapse.permanence + PERMANENCE_INC)
                else:
                    synapse.permanence = max(0.0, synapse.permanence - PERMANENCE_DEC)

    def update_segment_list(self, segment, active_synapses, sequence_segment_flag):           # Updates the segment list of a cell based on the current activity and learning    
        segment.synapses = active_synapses                                                    # This can involve creating new segments or adding synapses to existing segments
        segment.sequence_segment = sequence_segment_flag


    
    def learning_phase(self, active_columns, input_vector):                                   # Learning for Spatial Pooler
        for c in active_columns:
            for s in c.potential_synapses:
                if input_vector[s.source_input]:
                    s.permanence = min(1.0, s.permanence + PERMANENCE_INC)
                else:
                    s.permanence = max(0.0, s.permanence - PERMANENCE_DEC)
            c.connected_synapses = [s for s in c.potential_synapses if s.permanence > CONNECTED_PERM]

        
        inhibition_radius = self.average_receptive_field_size(self.columns)                    # Update inhibition radius based on average receptive field size 

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
    





    

# Example usage
input_space_size = 100
column_count = 2048
cells_per_column = 32
initial_synapses_per_column = 20

tp = TemporalPooler(input_space_size, column_count, cells_per_column, initial_synapses_per_column)
input_vector = np.random.randint(2, size=input_space_size)  # Example input vector (binary)
inhibition_radius = 10  # Example inhibition radius

# Spatial Pooling phases
active_columns = tp.compute_active_columns(input_vector, inhibition_radius)

# Temporal Pooling phases (use active_columns as input)
t = 0  # Initial time step
tp.compute_active_state(active_columns, t)                 # Compute the active state, predictive state, and update synapses
tp.compute_predictive_state(t)
tp.update_synapses(t)

# Learning phase for Spatial Pooler
tp.learning_phase(active_columns, input_vector)

# Increment time step
t += 1
