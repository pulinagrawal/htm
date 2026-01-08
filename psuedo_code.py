t = 0

for column in activeColumns:
    buPredicted = False
    lcChosen = False
    for cell in column.cells:
        if cell.prev_predictive:
            most_active_segment = cell.get_most_active_segment_from_prev_active_cells()
            if most_active_segment.sequenceSegment:
                buPredicted = True
                cell.set_active()
                if most_active_segment.is_active_from_prev_learnable_cells():
                    lcChosen = True
                    cell.set_learning()

    if column.bursting:
        for cell in column.cells:
            cell.set_active()

    if lcChosen == False:
        best_learning_cell = column.get_best_learning_cell()
        best_learning_cell.set_learning()
        best_learning_segment = best_learning_cell.get_best_learning_segment()
        best_learning_segment.grow_synapses_to_prev_learning_cells()
        best_learning_segment.sequence_segment = True


for cell in allCells:
    for segment in cell.segments:
        if segment.active:
            cell.set_predictive()

            segment.strengthen_current_active_synapses()

            best_learning_segment = cell.get_best_learning_segment()
            best_learning_segment.strengthen()
            best_learning_segment.grow_synapses_to_prev_learning_cells()

for cell in allCells:
    if cell.prev_predictive and not cell.active:
        cell.get_most_active_segment_from_prev_active_cells().weaken()