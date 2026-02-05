"""Main HTM Visualizer application."""

from typing import Any, Callable

import numpy as np
import pyvista as pv

from .brain_renderer import BrainRenderer
from .connection_renderer import ConnectionRenderer
from .controls import PlaybackController, setup_key_bindings
from .history import History, HTMSnapshot
from .colors import (
    color_to_float, TEXT_COLOR, TITLE_COLOR, BG_COLOR,
    COLORS, SEGMENT_COLORS,
)
from .brain_renderer import OUTGOING_SYN_COLOR, INCOMING_SYN_COLOR


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

        # Selection history
        self._selection_history: list[list[dict]] = []
        self._sel_hist_pos: int = -1

        # Legend and shortcuts visibility
        self._show_legend = False
        self._show_shortcuts = False

        # Field visibility - set of field names that are hidden
        self.hidden_fields: set[str] = set()

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

        # Selection via left-click: use iren click observer for ray-based picking
        self.plotter.iren.track_click_position(
            callback=self._on_click, side="left",
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

    def _on_click(self, click_pos):
        """Handle left-click via ray-based picking for accurate selection."""
        if click_pos is None:
            return

        # Build a ray from camera through the clicked world position
        cam_pos = np.array(self.plotter.camera.position)
        click_world = np.array(click_pos)
        ray_dir = click_world - cam_pos
        ray_len = np.linalg.norm(ray_dir)
        if ray_len < 1e-9:
            return
        ray_dir = ray_dir / ray_len

        info = self.brain_renderer.pick_by_ray(cam_pos, ray_dir)
        if info is None:
            return

        # Check if shift is held
        shift_held = False
        iren = self.plotter.iren.interactor
        if iren is not None:
            shift_held = bool(iren.GetShiftKey())

        if shift_held:
            existing = self._find_matching_selection(info)
            if existing is not None:
                self._selections.remove(existing)
            else:
                self._selections.append(info)
        else:
            self._selections = [info]

        self._push_selection_history()
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
        self._push_selection_history()
        self.brain_renderer.render_selection_highlights(self.plotter, self._selections)
        self._update_selection_overlay()
        self.plotter.render()

    def _push_selection_history(self):
        # Truncate forward history if we navigated back
        if self._sel_hist_pos < len(self._selection_history) - 1:
            self._selection_history = self._selection_history[:self._sel_hist_pos + 1]
        self._selection_history.append(list(self._selections))
        self._sel_hist_pos = len(self._selection_history) - 1
        # Cap at 100 entries
        if len(self._selection_history) > 100:
            self._selection_history.pop(0)
            self._sel_hist_pos = len(self._selection_history) - 1

    def selection_back(self):
        if self._sel_hist_pos > 0:
            self._sel_hist_pos -= 1
            self._selections = list(self._selection_history[self._sel_hist_pos])
            self.brain_renderer.render_selection_highlights(self.plotter, self._selections)
            self._update_selection_overlay()
            self.plotter.render()

    def selection_forward(self):
        if self._sel_hist_pos < len(self._selection_history) - 1:
            self._sel_hist_pos += 1
            self._selections = list(self._selection_history[self._sel_hist_pos])
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

    def toggle_outgoing_synapses(self):
        self.brain_renderer.show_outgoing_synapses = not self.brain_renderer.show_outgoing_synapses
        self._update_display()

    def toggle_incoming_synapses(self):
        self.brain_renderer.show_incoming_synapses = not self.brain_renderer.show_incoming_synapses
        self._update_display()

    def toggle_legend(self):
        self._show_legend = not self._show_legend
        self._update_legend()
        self.plotter.render()

    def toggle_shortcuts(self):
        self._show_shortcuts = not self._show_shortcuts
        self._update_shortcuts()
        self.plotter.render()

    def toggle_field(self, field_name: str):
        """Toggle visibility of a specific field."""
        if field_name in self.hidden_fields:
            self.hidden_fields.remove(field_name)
        else:
            self.hidden_fields.add(field_name)
        self.brain_renderer.set_hidden_fields(self.hidden_fields)
        self._update_display()

    def get_field_names(self) -> list[str]:
        """Return list of all field names."""
        return list(self.brain._input_fields.keys()) + list(self.brain._column_fields.keys())

    def cycle_field_visibility(self):
        """Cycle through fields to toggle visibility (for quick keyboard access)."""
        field_names = self.get_field_names()
        if not field_names:
            return
        # Find first visible field and hide it, or show all if all hidden
        visible = [f for f in field_names if f not in self.hidden_fields]
        if len(visible) == len(field_names):
            # All visible - hide the first one
            self.toggle_field(field_names[0])
        elif len(visible) == 0:
            # All hidden - show all
            self.hidden_fields.clear()
            self.brain_renderer.set_hidden_fields(self.hidden_fields)
            self._update_display()
        else:
            # Some hidden - hide next visible one
            self.toggle_field(visible[0])

    # ------------------------------------------------------------------
    # UI elements
    # ------------------------------------------------------------------

    def _add_title(self):
        self.plotter.add_text(
            self.title, position="upper_left", font_size=16,
            color=color_to_float(TITLE_COLOR), name="title",
        )

    def _add_controls_text(self):
        # Show minimal hint at top
        self.plotter.add_text(
            "Press H for keyboard shortcuts",
            position="upper_edge", font_size=9,
            color=(0.5, 0.5, 0.5), name="controls_hint",
        )
        self._shortcuts_actors = []

    def _update_shortcuts(self):
        # Remove existing shortcuts actors
        if hasattr(self, '_shortcuts_actors'):
            for actor in self._shortcuts_actors:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
        self._shortcuts_actors = []

        if not self._show_shortcuts:
            return

        # Build shortcuts text with fixed-width columns
        shortcuts_text = (
            "SPACE       Play/Pause    \n"
            "Arrow       Step fwd/back \n"
            "S           Synapses      \n"
            "P           Proximal      \n"
            "O           Outgoing      \n"
            "I           Incoming      \n"
            "R           Reset camera  \n"
            "L           Legend        \n"
            "H           Shortcuts     \n"
            "1-9         Toggle fields \n"
            "[  ]        Sel history   \n"
            "Click       Select        \n"
            "Shift+Click Multi-select  \n"
            "ESC         Clear select  "
        )

        # Create text actor with monospace font
        actor = self.plotter.add_text(
            shortcuts_text,
            position="upper_right",
            font_size=9,
            color=color_to_float(TEXT_COLOR),
            name="shortcuts_box",
            font="courier",
        )
        # Set text properties for box styling
        actor.GetTextProperty().SetBackgroundColor(0.1, 0.1, 0.1)
        actor.GetTextProperty().SetBackgroundOpacity(0.9)
        actor.GetTextProperty().SetFrameColor(0.3, 0.3, 0.3)
        actor.GetTextProperty().SetFrame(True)
        self._shortcuts_actors.append(actor)

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
        lines.append(f"Outgoing: {'ON' if self.brain_renderer.show_outgoing_synapses else 'OFF'}")
        lines.append(f"Incoming: {'ON' if self.brain_renderer.show_incoming_synapses else 'OFF'}")

        # Field visibility status
        field_names = self.get_field_names()
        if field_names:
            lines.append("\nFields (1-9 toggle):")
            for i, name in enumerate(field_names[:9]):
                visible = "✓" if name not in self.hidden_fields else "✗"
                lines.append(f"  {i+1}: {name} [{visible}]")

        return "\n".join(lines)

    def _add_widgets(self):
        def speed_callback(value):
            self.playback.speed_ms = int(value)

        self.plotter.add_slider_widget(
            speed_callback, rng=[50, 2000], value=500,
            title="Speed (ms)", pointa=(0.7, 0.05), pointb=(0.95, 0.05),
            style="modern", color=color_to_float(TEXT_COLOR),
        )

    def _update_legend(self):
        # Remove existing legend actors
        if hasattr(self, '_legend_actors'):
            for actor in self._legend_actors:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
        self._legend_actors = []

        if not self._show_legend:
            return

        # Build legend entries as (label, color) tuples using actual color variables
        cell_entries = [
            ("Active", color_to_float(COLORS["active"])),
            ("Predictive", color_to_float(COLORS["predictive"])),
            ("Bursting", color_to_float(COLORS["bursting"])),
            ("Winner", color_to_float(COLORS["winner"])),
            ("Correct Pred", color_to_float(COLORS["correct_prediction"])),
            ("Inactive", color_to_float(COLORS["inactive"])),
        ]

        segment_entries = [
            ("Seg Active", color_to_float(SEGMENT_COLORS["active"])),
            ("Seg Learning", color_to_float(SEGMENT_COLORS["learning"])),
            ("Seg Matching", color_to_float(SEGMENT_COLORS["matching"])),
            ("Seg Inactive", color_to_float(SEGMENT_COLORS["inactive"])),
        ]

        synapse_entries = [
            ("Outgoing Syn", color_to_float(OUTGOING_SYN_COLOR)),
            ("Incoming Syn", color_to_float(INCOMING_SYN_COLOR)),
        ]

        all_entries = cell_entries + segment_entries + synapse_entries

        legend_actor = self.plotter.add_legend(
            labels=all_entries,
            bcolor=(0.1, 0.1, 0.1),
            border=True,
            size=(0.15, 0.35),
            loc="lower left",
            name="legend",
        )
        self._legend_actors.append(legend_actor)

    def _reset_camera(self):
        self.plotter.camera_position = "xz"
        self.plotter.camera.azimuth = 20
        self.plotter.camera.elevation = 20
        self.plotter.reset_camera()
        self.plotter.camera.zoom(0.8)
