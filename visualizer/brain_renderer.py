"""Renders Brain state as 3D meshes using PyVista.

Cells are small spheres. Segments are small spheres offset radially from their
parent cell. Synapses are lines from segments to source cells, colored by
permanence (dark red -> dark green). Togglable via show_synapses flag.
"""

import math
import numpy as np
import pyvista as pv

from .colors import (
    COLORS, INPUT_FIELD_COLORS, SEGMENT_COLORS,
    color_to_float, state_color, segment_color, permanence_color,
    LABEL_COLOR,
)
from .history import HTMSnapshot

# Layout constants
COLUMN_SPACING = 1.0
CELL_SPACING = 0.6
LAYER_GAP = 5.0
CELL_RADIUS = 0.08
INPUT_CELL_RADIUS = 0.06
SEGMENT_RADIUS = 0.04
SEGMENT_OFFSET = 0.2

# Selection highlight
HIGHLIGHT_COLOR = (255, 255, 255)
HIGHLIGHT_RADIUS_SCALE = 2.5  # multiplier over base radius
HIGHLIGHT_OPACITY = 0.25

# Selection synapse colors
OUTGOING_SYN_COLOR = (255, 200, 50)   # gold - synapses FROM selected cell's segments
INCOMING_SYN_COLOR = (50, 200, 255)   # cyan - synapses TO selected cell


def grid_position(index: int, total: int) -> tuple[float, float]:
    cols = max(1, int(math.ceil(math.sqrt(total))))
    row = index // cols
    col = index % cols
    x = (col - cols / 2) * COLUMN_SPACING
    y = (row - (total // cols) / 2) * COLUMN_SPACING
    return x, y


def _segment_offset_direction(seg_idx: int, total_segs: int) -> np.ndarray:
    if total_segs <= 1:
        return np.array([SEGMENT_OFFSET, 0, 0])
    angle = 2 * math.pi * seg_idx / total_segs
    return np.array([
        SEGMENT_OFFSET * math.cos(angle),
        SEGMENT_OFFSET * math.sin(angle),
        0,
    ])


class FieldLayout:
    def __init__(self, name: str, base_z: float):
        self.name = name
        self.base_z = base_z
        self.cell_positions: dict[tuple[int, int], np.ndarray] = {}
        self.column_positions: dict[int, np.ndarray] = {}


class BrainRenderer:
    """Manages 3D rendering of Brain fields with cells, segments, and synapses."""

    def __init__(self, brain):
        self.brain = brain
        self.layouts: dict[str, FieldLayout] = {}
        self._cell_id_to_pos: dict[int, np.ndarray] = {}
        self.show_synapses = True
        self.show_outgoing_synapses = True
        self.show_incoming_synapses = True
        self.hidden_fields: set[str] = set()
        self._compute_layouts()
        self._build_cell_index()

    def _compute_layouts(self):
        z = 0.0
        for i, (name, field) in enumerate(self.brain._input_fields.items()):
            layout = FieldLayout(name, z)
            for ci in range(len(field.cells)):
                x, y = grid_position(ci, len(field.cells))
                layout.cell_positions[(ci, 0)] = np.array([x, y, z])
            self.layouts[name] = layout
        z += LAYER_GAP

        for name, field in self.brain._column_fields.items():
            layout = FieldLayout(name, z)
            n_cols = len(field.columns)
            for ci, col in enumerate(field.columns):
                x, y = grid_position(ci, n_cols)
                layout.column_positions[ci] = np.array([x, y, z])
                for ji in range(len(col.cells)):
                    layout.cell_positions[(ci, ji)] = np.array([x, y, z + ji * CELL_SPACING])
            self.layouts[name] = layout
            z += LAYER_GAP + len(field.columns[0].cells) * CELL_SPACING if field.columns else LAYER_GAP

    def _build_cell_index(self):
        self._cell_id_to_pos.clear()
        for name, field in self.brain._input_fields.items():
            for i, cell in enumerate(field.cells):
                self._cell_id_to_pos[id(cell)] = self.layouts[name].cell_positions[(i, 0)]
        for name, field in self.brain._column_fields.items():
            for ci, col in enumerate(field.columns):
                for ji, cell in enumerate(col.cells):
                    self._cell_id_to_pos[id(cell)] = self.layouts[name].cell_positions[(ci, ji)]

    def get_cell_position(self, field_name: str, col_idx: int, cell_idx: int = 0) -> np.ndarray:
        return self.layouts[field_name].cell_positions.get((col_idx, cell_idx), np.zeros(3))

    def set_hidden_fields(self, hidden: set[str]):
        """Update the set of hidden field names."""
        self.hidden_fields = hidden

    def is_field_visible(self, field_name: str) -> bool:
        """Check if a field should be rendered."""
        return field_name not in self.hidden_fields

    # ------------------------------------------------------------------
    # Initial render
    # ------------------------------------------------------------------

    def render_initial(self, plotter: pv.Plotter):
        for i, (name, field) in enumerate(self.brain._input_fields.items()):
            layout = self.layouts[name]
            color = INPUT_FIELD_COLORS[i % len(INPUT_FIELD_COLORS)]
            self._render_input_field(plotter, name, field, layout, color)

        for name, field in self.brain._column_fields.items():
            layout = self.layouts[name]
            self._render_column_field(plotter, name, field, layout)

    def _render_input_field(self, plotter, name, field, layout, base_color):
        n = len(field.cells)
        if n == 0:
            return
        
        # If field is hidden, render empty meshes
        if not self.is_field_visible(name):
            self._add_empty_mesh(plotter, f"input_{name}")
            label_pos = np.mean([layout.cell_positions[(i, 0)] for i in range(n)], axis=0) + np.array([0, 0, 1.0])
            plotter.add_point_labels(
                [label_pos], [""],
                font_size=14, text_color=color_to_float(LABEL_COLOR),
                shape=None, show_points=False, name=f"label_{name}",
            )
            return
            
        points = np.array([layout.cell_positions[(i, 0)] for i in range(n)])
        colors = np.zeros((n, 3), dtype=np.uint8)
        dim_color = tuple(c // 4 for c in base_color)
        for i, cell in enumerate(field.cells):
            colors[i] = base_color if cell.active else dim_color
        self._add_sphere_glyph(plotter, points, colors, INPUT_CELL_RADIUS, f"input_{name}")

        label_pos = np.mean(points, axis=0) + np.array([0, 0, 1.0])
        plotter.add_point_labels(
            [label_pos], [name.upper()],
            font_size=14, text_color=color_to_float(LABEL_COLOR),
            shape=None, show_points=False, name=f"label_{name}",
        )

    def _render_column_field(self, plotter, name, field, layout):
        # If field is hidden, render empty meshes
        if not self.is_field_visible(name):
            self._add_empty_mesh(plotter, f"cells_{name}")
            self._add_empty_mesh(plotter, f"segments_{name}")
            self._add_empty_mesh(plotter, f"synapses_{name}")
            if field.columns:
                all_z = [layout.cell_positions[(ci, ji)][2]
                         for ci in range(len(field.columns))
                         for ji in range(len(field.columns[0].cells))]
                center_xy = np.mean(
                    [layout.cell_positions[(ci, 0)][:2] for ci in range(len(field.columns))], axis=0
                )
                plotter.add_point_labels(
                    [np.array([center_xy[0], center_xy[1], max(all_z) + 1.5])],
                    [""],
                    font_size=16, text_color=color_to_float(LABEL_COLOR),
                    shape=None, show_points=False, name=f"label_{name}",
                )
            return
            
        self._render_cells(plotter, name, field, layout)
        self._render_segments_and_synapses(plotter, name, field, layout)
        # Label
        if field.columns:
            all_z = [layout.cell_positions[(ci, ji)][2]
                     for ci in range(len(field.columns))
                     for ji in range(len(field.columns[0].cells))]
            center_xy = np.mean(
                [layout.cell_positions[(ci, 0)][:2] for ci in range(len(field.columns))], axis=0
            )
            plotter.add_point_labels(
                [np.array([center_xy[0], center_xy[1], max(all_z) + 1.5])],
                [name.upper()],
                font_size=16, text_color=color_to_float(LABEL_COLOR),
                shape=None, show_points=False, name=f"label_{name}",
            )

    def _render_cells(self, plotter, name, field, layout):
        positions = []
        colors = []
        for ci, col in enumerate(field.columns):
            for ji, cell in enumerate(col.cells):
                positions.append(layout.cell_positions[(ci, ji)])
                colors.append(state_color(cell, col))
        if positions:
            self._add_sphere_glyph(
                plotter, np.array(positions), np.array(colors, dtype=np.uint8),
                CELL_RADIUS, f"cells_{name}",
            )

    def _render_segments_and_synapses(self, plotter, name, field, layout):
        seg_positions = []
        seg_colors = []
        syn_starts = []
        syn_ends = []
        syn_colors = []

        for ci, col in enumerate(field.columns):
            for ji, cell in enumerate(col.cells):
                if not cell.segments:
                    continue
                cell_pos = layout.cell_positions[(ci, ji)]
                n_segs = len(cell.segments)
                for si, seg in enumerate(cell.segments):
                    offset = _segment_offset_direction(si, n_segs)
                    seg_pos = cell_pos + offset
                    seg_positions.append(seg_pos)
                    seg_colors.append(segment_color(seg))

                    if self.show_synapses:
                        for syn in seg.synapses:
                            src_pos = self._cell_id_to_pos.get(id(syn.source_cell))
                            if src_pos is not None:
                                syn_starts.append(seg_pos)
                                syn_ends.append(src_pos)
                                syn_colors.append(permanence_color(syn.permanence))

        if seg_positions:
            self._add_cube_glyph(
                plotter, np.array(seg_positions),
                np.array(seg_colors, dtype=np.uint8),
                SEGMENT_RADIUS, f"segments_{name}",
            )
        else:
            # Clear old segments actor
            self._add_empty_mesh(plotter, f"segments_{name}")

        if syn_starts:
            self._draw_lines(
                plotter, np.array(syn_starts), np.array(syn_ends),
                np.array(syn_colors, dtype=np.uint8), f"synapses_{name}",
            )
        else:
            self._add_empty_mesh(plotter, f"synapses_{name}")

    # ------------------------------------------------------------------
    # Update methods
    # ------------------------------------------------------------------

    def update_live(self, plotter: pv.Plotter):
        for i, (name, field) in enumerate(self.brain._input_fields.items()):
            layout = self.layouts[name]
            n = len(field.cells)
            if n == 0:
                continue
            
            # Handle hidden fields
            if not self.is_field_visible(name):
                self._add_empty_mesh(plotter, f"input_{name}")
                continue
                
            base_color = INPUT_FIELD_COLORS[i % len(INPUT_FIELD_COLORS)]
            dim_color = tuple(c // 4 for c in base_color)
            points = np.array([layout.cell_positions[(ci, 0)] for ci in range(n)])
            colors = np.zeros((n, 3), dtype=np.uint8)
            for ci, cell in enumerate(field.cells):
                if cell.predictive and cell.active:
                    colors[ci] = COLORS["correct_prediction"]
                elif cell.predictive:
                    colors[ci] = COLORS["predictive"]
                elif cell.active:
                    colors[ci] = base_color
                else:
                    colors[ci] = dim_color
            self._add_sphere_glyph(plotter, points, colors, INPUT_CELL_RADIUS, f"input_{name}")

        for name, field in self.brain._column_fields.items():
            layout = self.layouts[name]
            # Handle hidden fields
            if not self.is_field_visible(name):
                self._add_empty_mesh(plotter, f"cells_{name}")
                self._add_empty_mesh(plotter, f"segments_{name}")
                self._add_empty_mesh(plotter, f"synapses_{name}")
                continue
            self._render_cells(plotter, name, field, layout)
            self._render_segments_and_synapses(plotter, name, field, layout)

    def update_from_snapshot(self, plotter: pv.Plotter, snapshot: HTMSnapshot):
        for i, (name, field) in enumerate(self.brain._input_fields.items()):
            layout = self.layouts[name]
            n = len(field.cells)
            if n == 0:
                continue
            
            # Handle hidden fields
            if not self.is_field_visible(name):
                self._add_empty_mesh(plotter, f"input_{name}")
                continue
                
            base_color = INPUT_FIELD_COLORS[i % len(INPUT_FIELD_COLORS)]
            dim_color = tuple(c // 4 for c in base_color)
            active_set = set(snapshot.input_active.get(name, []))
            pred_set = set(snapshot.input_predictive.get(name, []))
            points = np.array([layout.cell_positions[(ci, 0)] for ci in range(n)])
            colors = np.zeros((n, 3), dtype=np.uint8)
            for ci in range(n):
                if ci in pred_set and ci in active_set:
                    colors[ci] = COLORS["correct_prediction"]
                elif ci in pred_set:
                    colors[ci] = COLORS["predictive"]
                elif ci in active_set:
                    colors[ci] = base_color
                else:
                    colors[ci] = dim_color
            self._add_sphere_glyph(plotter, points, colors, INPUT_CELL_RADIUS, f"input_{name}")

        for name, field in self.brain._column_fields.items():
            layout = self.layouts[name]
            
            # Handle hidden fields
            if not self.is_field_visible(name):
                self._add_empty_mesh(plotter, f"cells_{name}")
                self._add_empty_mesh(plotter, f"segments_{name}")
                self._add_empty_mesh(plotter, f"synapses_{name}")
                continue
                
            active_set = set(snapshot.column_active_cells.get(name, []))
            winner_set = set(snapshot.column_winner_cells.get(name, []))
            pred_set = set(snapshot.column_predictive_cells.get(name, []))
            burst_set = set(snapshot.column_bursting.get(name, []))

            positions = []
            colors = []
            for ci, col in enumerate(field.columns):
                for ji in range(len(col.cells)):
                    positions.append(layout.cell_positions[(ci, ji)])
                    key = (ci, ji)
                    if ci in burst_set and key in active_set:
                        color = COLORS["bursting"]
                    elif key in pred_set and key in active_set:
                        color = COLORS["correct_prediction"]
                    elif key in pred_set:
                        color = COLORS["predictive"]
                    elif key in winner_set:
                        color = COLORS["winner"]
                    elif key in active_set:
                        color = COLORS["active"]
                    else:
                        color = COLORS["inactive"]
                    colors.append(color)
            if positions:
                self._add_sphere_glyph(
                    plotter, np.array(positions),
                    np.array(colors, dtype=np.uint8),
                    CELL_RADIUS, f"cells_{name}",
                )
            self._render_segments_and_synapses(plotter, name, field, layout)

    # ------------------------------------------------------------------
    # Selection: highlight glow + synapse tracing
    # ------------------------------------------------------------------

    def render_selection_highlights(self, plotter: pv.Plotter, selections: list[dict]):
        """Render glow highlights around selected elements and trace their synapses.

        Each selection dict has keys: type, field, pos, obj, and
        col/cell/seg indices depending on type.
        """
        if not selections:
            self._add_empty_mesh(plotter, "sel_highlight")
            self._add_empty_mesh(plotter, "sel_synapses")
            return

        # Glow spheres: additive transparent bright spheres at selection positions
        glow_pts = []
        glow_radii = []
        for sel in selections:
            glow_pts.append(sel["pos"])
            if sel["type"] == "segment":
                glow_radii.append(SEGMENT_RADIUS * HIGHLIGHT_RADIUS_SCALE)
            elif sel["type"] == "input_cell":
                glow_radii.append(INPUT_CELL_RADIUS * HIGHLIGHT_RADIUS_SCALE)
            else:
                glow_radii.append(CELL_RADIUS * HIGHLIGHT_RADIUS_SCALE)

        # Render each glow sphere individually so we can vary radius
        glow_meshes = pv.MultiBlock()
        for pt, r in zip(glow_pts, glow_radii):
            glow_meshes.append(pv.Sphere(radius=r, center=pt, theta_resolution=16, phi_resolution=16))
        merged = glow_meshes.combine()
        plotter.add_mesh(
            merged, color=color_to_float(HIGHLIGHT_COLOR),
            opacity=HIGHLIGHT_OPACITY, name="sel_highlight",
        )

        # Trace incoming and outgoing synapses for selected cells/segments
        selected_cell_ids = set()
        selected_seg_positions = {}  # id(seg) -> pos

        for sel in selections:
            if sel["type"] == "cell":
                selected_cell_ids.add(id(sel["obj"]))
            elif sel["type"] == "segment":
                selected_cell_ids.add(id(sel["obj"].parent_cell))
                selected_seg_positions[id(sel["obj"])] = sel["pos"]
            elif sel["type"] == "input_cell":
                selected_cell_ids.add(id(sel["obj"]))

        syn_starts = []
        syn_ends = []
        syn_colors = []

        # Walk all segments to find connections involving selected cells
        for name, field in self.brain._column_fields.items():
            layout = self.layouts[name]
            for ci, col in enumerate(field.columns):
                for ji, cell in enumerate(col.cells):
                    cell_pos = layout.cell_positions[(ci, ji)]
                    n_segs = len(cell.segments)
                    for si, seg in enumerate(cell.segments):
                        offset = _segment_offset_direction(si, n_segs)
                        seg_pos = cell_pos + offset

                        cell_is_selected = id(cell) in selected_cell_ids
                        seg_is_selected = id(seg) in selected_seg_positions

                        for syn in seg.synapses:
                            src_pos = self._cell_id_to_pos.get(id(syn.source_cell))
                            if src_pos is None:
                                continue

                            source_is_selected = id(syn.source_cell) in selected_cell_ids

                            if (cell_is_selected or seg_is_selected) and self.show_outgoing_synapses:
                                # Outgoing: this cell's segment reads from source
                                syn_starts.append(seg_pos)
                                syn_ends.append(src_pos)
                                syn_colors.append(OUTGOING_SYN_COLOR)

                            elif source_is_selected and self.show_incoming_synapses:
                                # Incoming: some other segment reads from selected cell
                                syn_starts.append(seg_pos)
                                syn_ends.append(src_pos)
                                syn_colors.append(INCOMING_SYN_COLOR)

        if syn_starts:
            self._draw_lines(
                plotter, np.array(syn_starts), np.array(syn_ends),
                np.array(syn_colors, dtype=np.uint8),
                "sel_synapses", line_width=3, opacity=0.9,
            )
        else:
            self._add_empty_mesh(plotter, "sel_synapses")

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------

    @staticmethod
    def _point_ray_metrics(point: np.ndarray, ray_origin: np.ndarray,
                           ray_dir: np.ndarray) -> tuple[float, float]:
        """Return (perpendicular distance to ray, depth along ray).

        Depth is used for tie-breaking: when two elements have similar
        ray distance, prefer the one closer to the camera.
        """
        v = point - ray_origin
        t = np.dot(v, ray_dir)
        if t < 0:
            return float("inf"), float("inf")
        closest = ray_origin + t * ray_dir
        return float(np.linalg.norm(point - closest)), t

    def pick_by_ray(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                    tolerance: float = 0.3) -> dict | None:
        """Find the element whose center is closest to the pick ray.

        Uses perpendicular distance for accuracy. When two elements are
        within 20% of each other in ray distance, the one closer to the
        camera (smaller depth) wins.
        """
        best_dist = tolerance
        best_depth = float("inf")
        result = None

        def _consider(dist, depth, info):
            nonlocal best_dist, best_depth, result
            # Strict improvement in ray distance
            if dist < best_dist * 0.8:
                best_dist = dist
                best_depth = depth
                result = info
            # Similar ray distance: prefer closer to camera
            elif dist < best_dist and depth < best_depth:
                best_dist = dist
                best_depth = depth
                result = info

        for name, field in self.brain._column_fields.items():
            layout = self.layouts[name]
            for ci, col in enumerate(field.columns):
                for ji, cell in enumerate(col.cells):
                    cell_pos = layout.cell_positions[(ci, ji)]
                    dist, depth = self._point_ray_metrics(cell_pos, ray_origin, ray_dir)
                    _consider(dist, depth, {
                        "type": "cell", "field": name,
                        "col": ci, "cell": ji, "obj": cell,
                        "pos": cell_pos, "segments": len(cell.segments),
                        "active": cell.active, "predictive": cell.predictive,
                        "winner": cell.winner,
                    })
                    n_segs = len(cell.segments)
                    for si, seg in enumerate(cell.segments):
                        offset = _segment_offset_direction(si, n_segs)
                        seg_pos = cell_pos + offset
                        dist, depth = self._point_ray_metrics(seg_pos, ray_origin, ray_dir)
                        _consider(dist, depth, {
                            "type": "segment", "field": name,
                            "col": ci, "cell": ji, "seg": si,
                            "obj": seg, "pos": seg_pos,
                            "synapses": len(seg.synapses),
                            "active": seg.active, "learning": seg.learning,
                            "matching": seg.matching,
                        })

        for name, field in self.brain._input_fields.items():
            layout = self.layouts[name]
            for ci, cell in enumerate(field.cells):
                cell_pos = layout.cell_positions[(ci, 0)]
                dist, depth = self._point_ray_metrics(cell_pos, ray_origin, ray_dir)
                _consider(dist, depth, {
                    "type": "input_cell", "field": name,
                    "index": ci, "obj": cell, "pos": cell_pos,
                    "active": cell.active, "predictive": cell.predictive,
                })

        return result

    def pick_info(self, pos: np.ndarray, tolerance: float = 0.3) -> dict | None:
        """Legacy point-based pick (kept for compatibility)."""
        best_dist = tolerance
        result = None
        for name, field in self.brain._column_fields.items():
            layout = self.layouts[name]
            for ci, col in enumerate(field.columns):
                for ji, cell in enumerate(col.cells):
                    cell_pos = layout.cell_positions[(ci, ji)]
                    dist = np.linalg.norm(pos - cell_pos)
                    if dist < best_dist:
                        best_dist = dist
                        result = {"type": "cell", "field": name, "col": ci,
                                  "cell": ji, "obj": cell, "pos": cell_pos,
                                  "segments": len(cell.segments),
                                  "active": cell.active, "predictive": cell.predictive,
                                  "winner": cell.winner}
        for name, field in self.brain._input_fields.items():
            layout = self.layouts[name]
            for ci, cell in enumerate(field.cells):
                cell_pos = layout.cell_positions[(ci, 0)]
                dist = np.linalg.norm(pos - cell_pos)
                if dist < best_dist:
                    best_dist = dist
                    result = {"type": "input_cell", "field": name, "index": ci,
                              "obj": cell, "pos": cell_pos,
                              "active": cell.active, "predictive": cell.predictive}
        return result

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _add_sphere_glyph(self, plotter, points, colors, radius, actor_name):
        cloud = pv.PolyData(points)
        cloud["colors"] = colors
        sphere_geom = pv.Sphere(radius=radius, theta_resolution=8, phi_resolution=8)
        spheres = cloud.glyph(geom=sphere_geom, orient=False, scale=False)
        spheres["colors"] = np.repeat(colors, sphere_geom.n_points, axis=0)
        plotter.add_mesh(
            spheres, scalars="colors", rgb=True,
            show_scalar_bar=False, name=actor_name, pickable=True,
        )

    def _add_cube_glyph(self, plotter, points, colors, size, actor_name):
        cloud = pv.PolyData(points)
        cloud["colors"] = colors
        cube_geom = pv.Cube(x_length=size * 2, y_length=size * 2, z_length=size * 2)
        cubes = cloud.glyph(geom=cube_geom, orient=False, scale=False)
        cubes["colors"] = np.repeat(colors, cube_geom.n_points, axis=0)
        plotter.add_mesh(
            cubes, scalars="colors", rgb=True,
            show_scalar_bar=False, name=actor_name, pickable=True,
        )

    def _draw_lines(self, plotter, starts, ends, colors, actor_name,
                    line_width=2, opacity=0.7):
        n = len(starts)
        points = np.empty((n * 2, 3))
        points[0::2] = starts
        points[1::2] = ends
        line_indices = np.empty(n * 3, dtype=np.int_)
        idx = np.arange(n)
        line_indices[0::3] = 2
        line_indices[1::3] = idx * 2
        line_indices[2::3] = idx * 2 + 1
        mesh = pv.PolyData(points, lines=line_indices)
        point_colors = np.repeat(colors, 2, axis=0)
        mesh["colors"] = point_colors
        plotter.add_mesh(
            mesh, scalars="colors", rgb=True,
            show_scalar_bar=False, line_width=line_width,
            opacity=opacity, name=actor_name,
        )

    def _add_empty_mesh(self, plotter, actor_name):
        """Add a tiny invisible mesh to clear a named actor."""
        plotter.add_mesh(
            pv.PolyData([0.0, 0.0, 0.0]), opacity=0.0, name=actor_name,
        )
