# HTM Brain Visualizer Requirements Specification (v2)

## 1. Overview
Build an interactive 3D visualization tool for HTM `Brain` instances containing multiple `InputField` and `ColumnField` objects. The visualizer displays the complete brain architecture with all fields, their cells, columns, segments, synapses, and inter-field connections.

## 2. Core Data Structures (from HTM.py)

| Entity | Key Attributes | Visualization Role |
|--------|----------------|-------------------|
| `Brain` | `fields` dict, `input_fields`, `column_fields` | Root container, field registry |
| `InputField` | `cells`, `encoder`, name (dict key) | Sensory input layer, encoding display |
| `ColumnField` | `columns`, `cells`, `input_fields`, `non_spatial`, `non_temporal` | Computational layer |
| `Column` | `cells`, `potential_synapses`, `connected_synapses`, `overlap`, states | Column unit with proximal connections |
| `Cell` | `parent_column`, `distal_field`, `segments`, states | Processing unit |
| `Segment` | `parent_cell`, `synapses`, `sequence_segment`, states | Dendrite with distal connections |
| `ProximalSynapse` / `DistalSynapse` | `source_cell`, `permanence` | Weighted connections |

## 3. Visual Layout

### 3.1 Brain-Level Layout
- Accept a `Brain` instance and automatically discover all fields via `brain.fields`
- Arrange `InputField` instances as flat grids at the bottom layer
- Stack `ColumnField` instances vertically above their input sources
- Display field names (from `Brain` dict keys) as floating labels
- Support user-configurable field positions (drag to reposition)

### 3.2 InputField Rendering
- Render as a 2D grid of cells (encoder output size determines dimensions)
- Color cells by active state (green=active, gray=inactive)
- Display encoded value and encoder type in label overlay
- Show decode preview for active patterns

### 3.3 ColumnField Rendering
- Render columns as vertical stacks of cells
- Arrange columns in a grid layout (e.g., 32×32 for 1024 columns)
- Distinguish `non_spatial` mode visually (direct input mapping indicator)
- Show aggregate layer stats: active count, bursting count, prediction accuracy

### 3.4 Connection Rendering
- **Proximal (Input→Column)**: Lines from `InputField` cells to column bases
  - Show only for selected column or on "show proximal" toggle
  - Color by permanence: red (<0.3) → yellow (0.3-0.5) → green (>0.5)
- **Distal (Cell→Cell)**: Lines between cells within same `ColumnField`
  - Show only for selected cell/segment
  - Trace back to source cells via `segment.synapses[].source_cell`

## 4. Color Coding Scheme

| State | Color | Applies To |
|-------|-------|------------|
| Active (current) | **Bright Green (#00FF00)** | Cells, Columns |
| Active (previous) | **Dim Green (#006600)** | Cells, Columns |
| Predictive (current) | **Magenta (#FF00FF)** | Cells |
| Predictive (previous) | **Dim Magenta (#660066)** | Cells |
| Bursting | **Red (#FF0000)** | Columns |
| Winner | **White (#FFFFFF)** | Cells |
| Learning segment | **Yellow (#FFFF00)** | Segments |
| Matching segment | **Orange (#FFA500)** | Segments |
| Connected synapse (≥0.5) | **Cyan (#00FFFF)** | Synapses |
| Disconnected synapse (<0.5) | **Red gradient** | Synapses |
| Inactive | **Dark Gray (#333333)** | All |

## 5. Interactive Features

### 5.1 Navigation
- Orbit camera (click-drag to rotate around scene center)
- Pan (middle-click or shift+drag)
- Zoom (scroll wheel)
- Focus on field (double-click field label to center view)
- Reset view hotkey (R)

### 5.2 Selection
- Click to select entities: Field, Column, Cell, Segment, Synapses
- Highlight selected entity with outline glow
- Show detailed info in side panel based on selection type
- Toggle visibility of connections for selected entity

#### 5.2.1 Selection Hierarchy
- Click `InputField` label → show all active cells, encoder info
- Click `ColumnField` label → show layer stats, highlight active columns
- Click Column → show proximal connections, column states
- Click Cell → show distal segments, cell states
- Click Segment → show all synapses with permanence values
- ESC to deselect, breadcrumb trail for navigation

### 5.3 Info Panel Contents
- **Field selected**: Name, type, cell count, active ratio, encoder params (if InputField)
- **Column selected**: Index, overlap, active/bursting/predictive states, cell count, duty cycle
- **Cell selected**: Index, parent column, active/winner/predictive states, segment count, duty cycle
- **Segment selected**: Index, synapse count (connected/total), active/learning/matching states, sequence_segment flag

## 6. Stepping & Playback

### 6.1 Controls
- Step forward: Call `brain.run(inputs)` with next input values
- Step backward: Restore from history buffer
- Auto-play with speed slider (1-60 steps/second)
- Pause/Resume toggle
- Input editor panel for manual value entry

### 6.2 Input Management
- Display all `InputField` names with current encoded value
- Allow editing values before stepping
- Support loading input sequences from CSV (like hot_gym.py)
- Show prediction vs actual comparison for each InputField

## 7. Debugging Features

### 7.1 Anomaly Detection Highlights
- Bursting columns (unpredicted input) - red outline
- False predictions (predictive but not activated) - striped magenta
- High overlap but not winner - yellow warning

### 7.2 Connection Tracing
- "Why active?" - trace back from active cell to input sources
- "What predicted?" - show segments that caused predictive state
- "Synapse health" - histogram of permanence distribution

### 7.3 Statistics Dashboard
- Mirror output from `brain.print_stats()` in UI panel
- Graphs: active columns over time, prediction accuracy, segment growth

## 8. Data Interface

### 8.1 Brain Integration
```python
from visualizer import HTMVisualizer

brain = Brain({
    "consumption": InputField(encoder_params=...),
    "date": InputField(encoder_params=...),
    "layer1": ColumnField(input_fields=[...]),
})

viz = HTMVisualizer(brain)
viz.run()  # Opens window, allows stepping
```

### 8.2 Callback Hooks
- `on_step(brain)` - called after each compute
- `on_select(entity)` - called when user selects element
- `on_input_change(field_name, value)` - called when user edits input

## 9. Technical Stack Recommendation

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **PyVista** | Native Python, good 3D, easy picking | Heavy dependency, limited UI | Good for core |
| **Vispy** | Fast, lightweight, GPU-accelerated | Lower-level API | Alternative |
| **Dear PyGui** | Excellent UI, fast, Python-native | 3D is basic | Best for UI |
| **PyVista + Dear PyGui** | Combine strengths | Integration complexity | **Recommended** |

## 10. File Structure

```
visualizer/
├── __init__.py
├── app.py                 # Main application, window management
├── brain_renderer.py      # Renders entire Brain structure
├── field_renderer.py      # InputField and ColumnField rendering
├── entity_renderer.py     # Column, Cell, Segment, Synapse meshes
├── connection_renderer.py # Proximal and distal connection lines
├── camera.py              # Navigation controls
├── selection.py           # Click handling, selection state
├── ui/
│   ├── panels.py          # Info panel, stats dashboard
│   ├── controls.py        # Playback controls, input editor
│   └── colors.py          # Color scheme configuration
├── data_bridge.py         # Brain state extraction
└── history.py             # State history for stepping back
```

## 11. Further Considerations

1. **Multi-ColumnField support?** Current codebase has single ColumnField per Brain in examples. If you plan to add hierarchical layers (L2→L3→L4 as in reference images), should the visualizer support inter-ColumnField distal connections? **Recommendation: Design for it now**

2. **Performance threshold?** With 1024 columns × 32 cells = 32K cells, full rendering is feasible. At what scale should LOD kick in? **Recommendation: LOD at >50K cells**

3. **Field positioning?** Should fields auto-layout based on input_fields dependencies, or allow full manual positioning? **Recommendation: Auto-layout with manual override**

---

## Reference Images
See `visualizer_ideas/` folder for visual mockups showing:
- Multi-layer vertical stacking (VISUAL L2, L3, L4, L5)
- Lateral field placement (Striatum D1, D2)
- Connection line rendering with permanence coloring
- Stats panels and selection info displays