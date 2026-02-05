"""Renders proximal connections between input cells and columns."""

import numpy as np
import pyvista as pv

from .colors import permanence_color
from .brain_renderer import BrainRenderer


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

    def render_proximal(self, plotter: pv.Plotter, column_field_name: str,
                        active_only: bool = True, max_connections: int = 50000):
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
