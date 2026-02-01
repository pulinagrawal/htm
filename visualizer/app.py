"""Main HTM Visualizer application."""

from typing import Any, Callable

import numpy as np
import pyvista as pv

from .brain_renderer import BrainRenderer
from .connection_renderer import ConnectionRenderer
from .controls import PlaybackController, setup_key_bindings
from .history import History, HTMSnapshot
from .colors import color_to_float, TEXT_COLOR, TITLE_COLOR, BG_COLOR


class HTMVisualizer:
    """Interactive 3D HTM Brain visualizer using PyVista.

    Controls:
        SPACE       Play/Pause auto-stepping
        RIGHT       Step forward
        LEFT        Step back in history
        S           Toggle synapse line visibility
        P           Toggle proximal connection visibility
        R           Reset camera
        Click       Select element (cell / segment / input cell)
        Shift+Click Add element to multi-selection
        ESC         Clear selection
    """

    def __init__(self, brain, input_sequence: list[dict[str, Any]] | None = None,
                 step_fn: Callable | None = None, title: str = "HTM Visualizer"):
        self.brain = brain
        self.input_sequence = input_sequence or []
        self.step_fn = step_fn
        self.title = title

        self.timestep = 0
        self.history = History(max_size=1000)
        self.show_proximal = False
        self.learn = True

        # Multi-selection list
        self._selections: list[dict] = []

        # Metric tracking
        self.burst_history: list[int] = []
        self.error_history: list[float] = []

        # Renderers
        self.brain_renderer = BrainRenderer(brain)
        self.conn_renderer = ConnectionRenderer(self.brain_renderer)

        self.plotter = None

    def run(self):
        pv.global_theme.background = "black"
        pv.global_theme.font.color = "white"

        self.plotter = pv.Plotter(title=self.title, window_size=(1600, 900))
        self.plotter.set_background("black")
        self.plotter.enable_anti_aliasing("ssaa")

        self.brain_renderer.render_initial(self.plotter)

        self.playback = PlaybackController(
            step_callback=self._do_step,
            update_callback=self._update_display,
        )
        self.playback.step_back = self._step_back_history
        setup_key_bindings(self.plotter, self)

        # Picking: left-click selects, we check shift in the callback
        self.plotter.enable_point_picking(
            callback=self._on_pick, show_message=False,
            show_point=False, pickable_window=True,
            left_clicking=True, tolerance=0.025,
        )

        self._add_title()
        self._add_controls_text()
        self._add_stats_overlay()
        self._add_selection_overlay()
        self._add_widgets()

        self._reset_camera()
        self._capture_snapshot({})
        self.plotter.show()

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def _do_step(self):
        if self.step_fn:
            inputs = self.step_fn(self.timestep)
        elif self.timestep < len(self.input_sequence):
            inputs = self.input_sequence[self.timestep]
        elif self.input_sequence:
            inputs = self.input_sequence[self.timestep % len(self.input_sequence)]
        else:
            return

        try:
            predictions = self.brain.prediction()
        except Exception:
            predictions = {}

        self.brain.step(inputs, learn=self.learn)
        self.timestep += 1
        self._track_metrics(inputs, predictions)
        self._capture_snapshot(inputs, predictions)

    def _track_metrics(self, inputs, predictions):
        total_bursting = sum(
            len(f.bursting_columns) for f in self.brain._column_fields.values()
        )
        self.burst_history.append(total_bursting)
        for name in self.brain._input_fields:
            if name in predictions and name in inputs:
                pred, actual = predictions[name], inputs[name]
                if isinstance(pred, (int, float)) and isinstance(actual, (int, float)):
                    self.error_history.append(abs(pred - actual))
                break

    def _capture_snapshot(self, inputs, predictions=None):
        self.history.capture(self.brain, self.timestep, inputs, predictions or {})

    def _update_display(self):
        snap = self.history.current
        if snap:
            self.brain_renderer.update_from_snapshot(self.plotter, snap)
        else:
            self.brain_renderer.update_live(self.plotter)

        self.conn_renderer.clear(self.plotter)
        if self.show_proximal:
            for name in self.brain._column_fields:
                self.conn_renderer.render_proximal(self.plotter, name)

        # Selection highlights + synapse tracing
        self.brain_renderer.render_selection_highlights(self.plotter, self._selections)

        self._update_stats_overlay()
        self._update_selection_overlay()
        self.plotter.render()

    def _step_back_history(self):
        if self.history.can_step_back:
            self.history.step_back()
            snap = self.history.current
            if snap:
                self.brain_renderer.update_from_snapshot(self.plotter, snap)
                self.brain_renderer.render_selection_highlights(self.plotter, self._selections)
                self._update_stats_overlay()
                self.plotter.render()

    # ------------------------------------------------------------------
    # Selection (picking)
    # ------------------------------------------------------------------

    def _on_pick(self, point):
        """Handle click. Plain click = replace selection. Shift+click = add."""
        if point is None:
            return

        pos = np.array(point)
        info = self.brain_renderer.pick_info(pos)
        if info is None:
            return

        # Check if shift is held via the interactor
        shift_held = False
        iren = self.plotter.iren.interactor
        if iren is not None:
            shift_held = bool(iren.GetShiftKey())

        if shift_held:
            # Toggle: if already selected, deselect; otherwise add
            existing = self._find_matching_selection(info)
            if existing is not None:
                self._selections.remove(existing)
            else:
                self._selections.append(info)
        else:
            # Replace selection
            self._selections = [info]

        self.brain_renderer.render_selection_highlights(self.plotter, self._selections)
        self._update_selection_overlay()
        self.plotter.render()

    def _find_matching_selection(self, info: dict) -> dict | None:
        """Find an existing selection that matches the same element."""
        for sel in self._selections:
            if sel["type"] != info["type"]:
                continue
            if sel["type"] == "cell":
                if sel["field"] == info["field"] and sel["col"] == info["col"] and sel["cell"] == info["cell"]:
                    return sel
            elif sel["type"] == "segment":
                if sel["field"] == info["field"] and sel["col"] == info["col"] and sel["cell"] == info["cell"] and sel["seg"] == info["seg"]:
                    return sel
            elif sel["type"] == "input_cell":
                if sel["field"] == info["field"] and sel["index"] == info["index"]:
                    return sel
        return None

    def clear_selection(self):
        self._selections.clear()
        self.brain_renderer.render_selection_highlights(self.plotter, self._selections)
        self._update_selection_overlay()
        self.plotter.render()

    def _add_selection_overlay(self):
        self.plotter.add_text(
            "", position="lower_right", font_size=10,
            color=color_to_float(TEXT_COLOR), name="selection_info",
        )

    def _update_selection_overlay(self):
        if not self._selections:
            text = "Click to select | Shift+Click: multi-select"
        elif len(self._selections) == 1:
            text = self._format_selection(self._selections[0])
        else:
            parts = [f"Selected: {len(self._selections)} elements"]
            for i, sel in enumerate(self._selections):
                parts.append(f"\n--- [{i+1}] ---")
                parts.append(self._format_selection(sel))
            text = "\n".join(parts)

        self.plotter.add_text(
            text, position="lower_right", font_size=10,
            color=color_to_float(TEXT_COLOR), name="selection_info",
        )

    def _format_selection(self, sel: dict) -> str:
        if sel["type"] == "cell":
            cell = sel["obj"]
            text = (
                f"CELL  {sel['field']} col={sel['col']} cell={sel['cell']}\n"
                f"  active:     {sel['active']}\n"
                f"  predictive: {sel['predictive']}\n"
                f"  winner:     {sel['winner']}\n"
                f"  segments:   {sel['segments']}\n"
            )
            for si, seg in enumerate(cell.segments):
                connected = sum(1 for s in seg.synapses if s.permanence >= 0.5)
                text += (
                    f"  seg[{si}] syn={len(seg.synapses)} "
                    f"conn={connected} "
                    f"act={seg.active} lrn={seg.learning}\n"
                )
            return text
        elif sel["type"] == "segment":
            seg = sel["obj"]
            connected = sum(1 for s in seg.synapses if s.permanence >= 0.5)
            perms = [s.permanence for s in seg.synapses]
            text = (
                f"SEGMENT  {sel['field']} col={sel['col']} "
                f"cell={sel['cell']} seg={sel['seg']}\n"
                f"  synapses:   {sel['synapses']}\n"
                f"  connected:  {connected}\n"
                f"  active:     {sel['active']}\n"
                f"  learning:   {sel['learning']}\n"
                f"  matching:   {sel['matching']}\n"
            )
            if perms:
                text += (
                    f"  perm range: {min(perms):.3f} - {max(perms):.3f}\n"
                    f"  perm mean:  {sum(perms)/len(perms):.3f}\n"
                )
            return text
        elif sel["type"] == "input_cell":
            return (
                f"INPUT CELL  {sel['field']} idx={sel['index']}\n"
                f"  active:     {sel['active']}\n"
                f"  predictive: {sel['predictive']}\n"
            )
        return ""

    # ------------------------------------------------------------------
    # Public control methods (called by key bindings)
    # ------------------------------------------------------------------

    def toggle_play(self):
        self.playback.toggle_play(self.plotter)

    def step_forward(self):
        self._do_step()
        self._update_display()

    def step_back(self):
        self._step_back_history()

    def reset_view(self):
        self._reset_camera()

    def toggle_connections(self):
        pass

    def toggle_proximal(self):
        self.show_proximal = not self.show_proximal
        self._update_display()

    def toggle_synapses(self):
        self.brain_renderer.show_synapses = not self.brain_renderer.show_synapses
        self._update_display()

    # ------------------------------------------------------------------
    # UI elements
    # ------------------------------------------------------------------

    def _add_title(self):
        self.plotter.add_text(
            self.title, position="upper_left", font_size=16,
            color=color_to_float(TITLE_COLOR), name="title",
        )

    def _add_controls_text(self):
        self.plotter.add_text(
            "SPACE: Play/Pause  |  \u2192: Step  |  \u2190: Back  |  "
            "S: Synapses  |  P: Proximal  |  R: Reset  |  "
            "Click: Select  |  Shift+Click: Multi  |  ESC: Deselect",
            position="upper_edge", font_size=9,
            color=(0.5, 0.5, 0.5), name="controls_help",
        )

    def _add_stats_overlay(self):
        self.plotter.add_text(
            self._build_stats_text(), position="lower_left",
            font_size=10, color=color_to_float(TEXT_COLOR), name="stats",
        )

    def _update_stats_overlay(self):
        self.plotter.add_text(
            self._build_stats_text(), position="lower_left",
            font_size=10, color=color_to_float(TEXT_COLOR), name="stats",
        )

    def _build_stats_text(self) -> str:
        snap = self.history.current
        lines = [f"Timestep: {snap.timestep if snap else self.timestep}"]

        if snap:
            for name, val in snap.inputs.items():
                lines.append(f"  {name}: {val:.4f}" if isinstance(val, float) else f"  {name}: {val}")
            for name in self.brain._input_fields:
                pred = snap.predictions.get(name)
                if pred is not None:
                    lines.append(f"  pred({name}): {pred:.4f}" if isinstance(pred, float) else f"  pred({name}): {pred}")
            lines.append("")
            for name in self.brain._column_fields:
                n_active = len(snap.column_active_cols.get(name, []))
                n_burst = len(snap.column_bursting.get(name, []))
                n_pred = len(snap.column_predictive_cells.get(name, []))
                n_seg = snap.num_segments.get(name, 0)
                n_syn = snap.num_synapses.get(name, 0)
                lines.append(f"[{name}]")
                lines.append(f"  Active Cols: {n_active}")
                lines.append(f"  Bursting:    {n_burst}")
                lines.append(f"  Predictive:  {n_pred}")
                lines.append(f"  Segments:    {n_seg}")
                lines.append(f"  Synapses:    {n_syn}")

        if self.burst_history:
            recent = self.burst_history[-20:]
            lines.append(f"\nBurst Avg(20): {sum(recent)/len(recent):.1f}")
        if self.error_history:
            recent = self.error_history[-20:]
            lines.append(f"Error Avg(20): {sum(recent)/len(recent):.4f}")

        lines.append(f"\nHistory: {self.history._position + 1}/{len(self.history)}")
        lines.append(f"Synapses: {'ON' if self.brain_renderer.show_synapses else 'OFF'}")
        lines.append(f"Proximal: {'ON' if self.show_proximal else 'OFF'}")
        return "\n".join(lines)

    def _add_widgets(self):
        def speed_callback(value):
            self.playback.speed_ms = int(value)

        self.plotter.add_slider_widget(
            speed_callback, rng=[50, 2000], value=500,
            title="Speed (ms)", pointa=(0.7, 0.05), pointb=(0.95, 0.05),
            style="modern", color=color_to_float(TEXT_COLOR),
        )

    def _reset_camera(self):
        self.plotter.camera_position = "xz"
        self.plotter.camera.azimuth = 20
        self.plotter.camera.elevation = 20
        self.plotter.reset_camera()
        self.plotter.camera.zoom(0.8)
