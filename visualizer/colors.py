"""Color constants and utilities for HTM visualization."""

import numpy as np

# Cell state colors (RGB 0-255)
COLORS = {
    "inactive":    (40, 40, 40),
    "active":      (0, 255, 0),
    "predictive":  (200, 0, 255),
    "bursting":    (255, 50, 0),
    "winner":      (255, 255, 0),
    "correct_prediction": (0, 255, 255),
    "learning":    (255, 128, 0),
}

# Segment state colors
SEGMENT_COLORS = {
    "inactive":  (60, 60, 60),
    "active":    (0, 255, 100),
    "learning":  (255, 180, 0),
    "matching":  (100, 100, 255),
}

# Input field colors for multiple input fields
# Note: Avoid red (255, 50, 0) as it's used for bursting cells
INPUT_FIELD_COLORS = [
    (0, 200, 255),     # Cyan-blue
    (0, 255, 0),       # Green
    (255, 200, 0),     # Orange-yellow
    (255, 0, 255),     # Magenta
    (100, 255, 100),   # Light green
    (255, 150, 150),   # Light pink
]

# UI colors
BG_COLOR = (0, 0, 0)
TEXT_COLOR = (0, 255, 0)
TITLE_COLOR = (0, 255, 0)
STATS_COLOR = (0, 200, 0)
LABEL_COLOR = (255, 255, 0)

# Connection colors
PROXIMAL_COLOR = (100, 100, 100)
DISTAL_COLOR = (0, 200, 200)


def permanence_color(permanence: float) -> tuple:
    """Map permanence (0-1) to color gradient: dark red → light red → light green → dark green.

    0.0  = dark red   (128, 0, 0)
    0.25 = light red  (255, 100, 100)
    0.5  = neutral    (200, 200, 100)  (crossover at connected threshold)
    0.75 = light green (100, 255, 100)
    1.0  = dark green  (0, 128, 0)
    """
    if permanence <= 0.5:
        t = permanence / 0.5  # 0..1
        if t <= 0.5:
            # dark red (180,30,30) → bright red (255,100,100)
            s = t * 2
            return (int(180 + 75 * s), int(30 + 70 * s), int(30 + 70 * s))
        else:
            # bright red (255,100,100) → warm neutral (240,200,80)
            s = (t - 0.5) * 2
            return (int(255 - 15 * s), int(100 + 100 * s), int(100 - 20 * s))
    else:
        t = (permanence - 0.5) / 0.5  # 0..1
        if t <= 0.5:
            # warm neutral (240,200,80) → bright green (80,255,80)
            s = t * 2
            return (int(240 - 160 * s), int(200 + 55 * s), int(80 * (1 + s)))
        else:
            # bright green (80,255,80) → medium green (30,200,30)
            s = (t - 0.5) * 2
            return (int(80 - 50 * s), int(255 - 55 * s), int(80 - 50 * s))


def color_to_float(color: tuple) -> tuple:
    """Convert 0-255 RGB to 0-1 float RGB."""
    return (color[0] / 255, color[1] / 255, color[2] / 255)


# All toggleable cell state names
CELL_STATES = ["active", "predictive", "bursting", "winner", "correct_prediction"]


def state_color(cell, column=None, hidden_states: set | None = None) -> tuple:
    """Determine cell color based on its current state. Returns 0-255 RGB.

    Priority: correct_prediction > bursting > predictive > winner > active > inactive
    
    Args:
        cell: The cell object to get color for.
        column: The column containing the cell (optional, needed for bursting check).
        hidden_states: Set of state names whose coloring should be disabled.
    """
    hidden = hidden_states or set()
    
    if column and column.bursting and cell.active and "bursting" not in hidden:
        return COLORS["bursting"]
    if cell.predictive and cell.prev_predictive and cell.active and "correct_prediction" not in hidden:
        return COLORS["correct_prediction"]
    if cell.predictive and "predictive" not in hidden:
        return COLORS["predictive"]
    if cell.winner and "winner" not in hidden:
        return COLORS["winner"]
    if cell.active and "active" not in hidden:
        return COLORS["active"]
    return COLORS["inactive"]


def segment_color(segment) -> tuple:
    """Determine segment color based on state."""
    if segment.learning:
        return SEGMENT_COLORS["learning"]
    if segment.active:
        return SEGMENT_COLORS["active"]
    if segment.matching:
        return SEGMENT_COLORS["matching"]
    return SEGMENT_COLORS["inactive"]
