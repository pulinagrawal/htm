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
    "go_depolarized": (0, 180, 255),      # Blue - motivated/approach
    "nogo_depolarized": (255, 100, 50),    # Orange-red - inhibited/avoid
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
TEXT_COLOR = (255, 255, 255)
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


# OutputField probability gradient colors
PROB_NEUTRAL_COLOR = (80, 80, 80)       # At base probability — dim gray
PROB_GO_COLOR = (255, 200, 0)           # Above base — gold (Go excitation)
PROB_NOGO_COLOR = (80, 50, 180)         # Below base — purple (NoGo suppression)

# Apical segment state colors
APICAL_SEGMENT_COLORS = {
    "inactive":  (60, 60, 60),
    "go_active":    (0, 180, 255),    # Blue - Go segment active
    "nogo_active":  (255, 100, 50),   # Orange - NoGo segment active
    "learning":  (255, 180, 0),
}

# All toggleable cell state names
CELL_STATES = ["active", "predictive", "bursting", "winner", "correct_prediction",
               "go_depolarized", "nogo_depolarized"]

# All toggleable segment state names
SEGMENT_STATES = ["active", "learning", "matching"]


def state_color(cell, column=None, hidden_states: set | None = None) -> tuple:
    """Determine cell color based on its current state. Returns 0-255 RGB.

    Priority: correct_prediction > bursting > predictive > go/nogo > winner > active > inactive

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
    if hasattr(cell, 'go_depolarized') and cell.go_depolarized and "go_depolarized" not in hidden:
        return COLORS["go_depolarized"]
    if hasattr(cell, 'nogo_depolarized') and cell.nogo_depolarized and "nogo_depolarized" not in hidden:
        return COLORS["nogo_depolarized"]
    if cell.winner and "winner" not in hidden:
        return COLORS["winner"]
    if cell.active and "active" not in hidden:
        return COLORS["active"]
    return COLORS["inactive"]


def cell_active_states(cell, column=None, hidden_states: set | None = None) -> list[tuple[str, tuple]]:
    """Return ALL active states on a cell as [(state_name, color), ...].

    Unlike state_color() which returns only the highest-priority state,
    this collects every active state for wedge-based rendering.
    Respects hidden_states — hidden states are excluded.
    Falls back to [("inactive", COLORS["inactive"])] if none active.
    """
    hidden = hidden_states or set()
    states = []

    if column and column.bursting and cell.active and "bursting" not in hidden:
        states.append(("bursting", COLORS["bursting"]))
    if cell.predictive and cell.prev_predictive and cell.active and "correct_prediction" not in hidden:
        states.append(("correct_prediction", COLORS["correct_prediction"]))
    if cell.predictive and "predictive" not in hidden:
        states.append(("predictive", COLORS["predictive"]))
    if hasattr(cell, 'go_depolarized') and cell.go_depolarized and "go_depolarized" not in hidden:
        states.append(("go_depolarized", COLORS["go_depolarized"]))
    if hasattr(cell, 'nogo_depolarized') and cell.nogo_depolarized and "nogo_depolarized" not in hidden:
        states.append(("nogo_depolarized", COLORS["nogo_depolarized"]))
    if cell.winner and "winner" not in hidden:
        states.append(("winner", COLORS["winner"]))
    if cell.active and "active" not in hidden:
        states.append(("active", COLORS["active"]))

    return states if states else [("inactive", COLORS["inactive"])]


def cell_active_states_from_sets(
    key: tuple[int, int],
    col_idx: int,
    active_set: set,
    winner_set: set,
    pred_set: set,
    burst_set: set,
    go_set: set,
    nogo_set: set,
    hidden_states: set | None = None,
) -> list[tuple[str, tuple]]:
    """Return ALL active states for a cell using set-based lookups (snapshot path).

    Same logic as cell_active_states() but works with precomputed sets
    instead of cell objects.
    """
    hidden = hidden_states or set()
    states = []

    if col_idx in burst_set and key in active_set and "bursting" not in hidden:
        states.append(("bursting", COLORS["bursting"]))
    if key in pred_set and key in active_set and "correct_prediction" not in hidden:
        states.append(("correct_prediction", COLORS["correct_prediction"]))
    if key in pred_set and "predictive" not in hidden:
        states.append(("predictive", COLORS["predictive"]))
    if key in go_set and "go_depolarized" not in hidden:
        states.append(("go_depolarized", COLORS["go_depolarized"]))
    if key in nogo_set and "nogo_depolarized" not in hidden:
        states.append(("nogo_depolarized", COLORS["nogo_depolarized"]))
    if key in winner_set and "winner" not in hidden:
        states.append(("winner", COLORS["winner"]))
    if key in active_set and "active" not in hidden:
        states.append(("active", COLORS["active"]))

    return states if states else [("inactive", COLORS["inactive"])]


def segment_color(segment, hidden_states: set | None = None) -> tuple:
    """Determine segment color based on state.

    Args:
        segment: The segment object to get color for.
        hidden_states: Set of segment state names whose coloring should be disabled.
    """
    hidden = hidden_states or set()

    if segment.learning and "learning" not in hidden:
        return SEGMENT_COLORS["learning"]
    if segment.active and "active" not in hidden:
        return SEGMENT_COLORS["active"]
    if segment.matching and "matching" not in hidden:
        return SEGMENT_COLORS["matching"]
    return SEGMENT_COLORS["inactive"]


def apical_segment_color(segment) -> tuple:
    """Determine apical segment color based on sign and state."""
    if segment.learning:
        return APICAL_SEGMENT_COLORS["learning"]
    if segment.active:
        if segment.sign > 0:
            return APICAL_SEGMENT_COLORS["go_active"]
        else:
            return APICAL_SEGMENT_COLORS["nogo_active"]
    return APICAL_SEGMENT_COLORS["inactive"]


def probability_color(prob: float, base_prob: float) -> tuple:
    """Map activation probability to a diverging color relative to base probability.

    prob < base_prob  →  purple (NoGo suppression), darker as prob approaches 0
    prob == base_prob →  dim gray (neutral / unmodulated)
    prob > base_prob  →  gold (Go excitation), brighter as prob approaches 1

    Args:
        prob: The cell's activation probability [0, 1].
        base_prob: The field's base_activation_probability (the unmodulated default).
    """
    prob = max(0.0, min(1.0, prob))
    base_prob = max(0.001, min(0.999, base_prob))

    if prob <= base_prob:
        # Below base: blend from purple (at 0) to neutral gray (at base)
        t = prob / base_prob if base_prob > 0 else 0.0
        return (
            int(PROB_NOGO_COLOR[0] + (PROB_NEUTRAL_COLOR[0] - PROB_NOGO_COLOR[0]) * t),
            int(PROB_NOGO_COLOR[1] + (PROB_NEUTRAL_COLOR[1] - PROB_NOGO_COLOR[1]) * t),
            int(PROB_NOGO_COLOR[2] + (PROB_NEUTRAL_COLOR[2] - PROB_NOGO_COLOR[2]) * t),
        )
    else:
        # Above base: blend from neutral gray (at base) to bright gold (at 1)
        span = 1.0 - base_prob
        t = (prob - base_prob) / span if span > 0 else 0.0
        return (
            int(PROB_NEUTRAL_COLOR[0] + (PROB_GO_COLOR[0] - PROB_NEUTRAL_COLOR[0]) * t),
            int(PROB_NEUTRAL_COLOR[1] + (PROB_GO_COLOR[1] - PROB_NEUTRAL_COLOR[1]) * t),
            int(PROB_NEUTRAL_COLOR[2] + (PROB_GO_COLOR[2] - PROB_NEUTRAL_COLOR[2]) * t),
        )
