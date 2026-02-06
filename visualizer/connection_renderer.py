"""Renders proximal connections between input cells and columns."""

import numpy as np
import pyvista as pv

from .colors import permanence_color
from .brain_renderer import BrainRenderer

# Proximal synapse colors
PROXIMAL_OUTGOING_COLOR = (255, 200, 50)   # gold - synapses FROM selected column


class ConnectionRenderer:
    """Draws proximal connection lines (input → column base).

    Distal connections (segment → source cell) are handled directly by
    BrainRenderer since segments are always rendered with their synapses.
    """

    def __init__(self, brain_renderer: BrainRenderer):
        self.br = brain_renderer
        self.brain = brain_renderer.brain
        self._actors: dict[str, object] = {}
        self.show_all_potential = True  # Show potential synapses, not just connected
        self.show_proximal_synapses = True  # Toggle for proximal synapse visibility

    def render_proximal_for_selection(self, plotter: pv.Plotter, selections: list[dict]):
        """Render proximal connections only for selected columns.
        
        Selection is based on the bottom-most cell (cell index 0) of a column.
        """
        actor_name = "proximal_selection"
        
        if not self.show_proximal_synapses or not selections:
            if actor_name in self._actors:
                plotter.remove_actor(actor_name, render=False)
                del self._actors[actor_name]
            return

        starts = []
        ends = []
        colors = []

        # Find selected columns from cell selections
        selected_columns = []
        for sel in selections:
            if sel["type"] == "cell" and sel.get("cell") == 0:
                # Bottom-most cell selected - this represents the column
                field_name = sel["field"]
                col_idx = sel["col"]
                field = self.brain._column_fields.get(field_name)
                if field and col_idx < len(field.columns):
                    col = field.columns[col_idx]
                    col_pos = self.br.layouts[field_name].column_positions.get(col_idx)
                    if col_pos is not None:
                        selected_columns.append((col, col_pos))

        # Render proximal synapses for selected columns
        for col, col_pos in selected_columns:
            synapses = getattr(col, 'potential_synapses', []) if self.show_all_potential else getattr(col, 'connected_synapses', [])
            for syn in synapses:
                src_pos = self.br._cell_id_to_pos.get(id(syn.source_cell))
                if src_pos is not None:
                    starts.append(src_pos)
                    ends.append(col_pos)
                    colors.append(permanence_color(syn.permanence))

        if not starts:
            if actor_name in self._actors:
                plotter.remove_actor(actor_name, render=False)
                del self._actors[actor_name]
            return

        self.br._draw_lines(
            plotter, np.array(starts), np.array(ends),
            np.array(colors, dtype=np.uint8), actor_name,
            line_width=2, opacity=0.9,
        )
        self._actors[actor_name] = True

    def render_proximal(self, plotter: pv.Plotter, column_field_name: str,
                        active_only: bool = True, max_connections: int = 500000):
        """Render proximal connections from input cells to column bases."""
        field = self.brain._column_fields.get(column_field_name)
        if not field:
            return

        starts = []
        ends = []
        colors = []

        for ci, col in enumerate(field.columns):
            if active_only and not col.active:
                continue
            col_pos = self.br.layouts[column_field_name].column_positions.get(ci)
            if col_pos is None:
                continue

            # Use potential_synapses if show_all_potential, otherwise only connected
            synapses = getattr(col, 'potential_synapses', []) if self.show_all_potential else getattr(col, 'connected_synapses', [])
            for syn in synapses:
                src_pos = self.br._cell_id_to_pos.get(id(syn.source_cell))
                if src_pos is not None:
                    starts.append(src_pos)
                    ends.append(col_pos)
                    colors.append(permanence_color(syn.permanence))

        # Truncate if too many connections (for performance)
        if len(starts) > max_connections:
            starts = starts[:max_connections]
            ends = ends[:max_connections]
            colors = colors[:max_connections]

        actor_name = f"proximal_{column_field_name}"
        if not starts:
            # Remove existing if any
            if actor_name in self._actors:
                plotter.remove_actor(actor_name, render=False)
                del self._actors[actor_name]
            return

        self.br._draw_lines(
            plotter, np.array(starts), np.array(ends),
            np.array(colors, dtype=np.uint8), actor_name,
        )
        self._actors[actor_name] = True

    def clear(self, plotter: pv.Plotter):
        """Remove all proximal connection actors."""
        for name in list(self._actors):
            plotter.remove_actor(name, render=False)
        self._actors.clear()
