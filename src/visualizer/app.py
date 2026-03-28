"""Main HTM Visualizer application."""

from typing import Any, Callable, Iterable

import numpy as np
import pyvista as pv

from .brain_renderer import BrainRenderer
from .connection_renderer import ConnectionRenderer
from .controls import PlaybackController, setup_key_bindings
from .history import History, HTMSnapshot
from .mode_manager import Mode, ModeManager, MODE_COLORS, MODE_LABELS
from .colors import (
    color_to_float, TEXT_COLOR, TITLE_COLOR, BG_COLOR,
    COLORS, SEGMENT_COLORS, APICAL_SEGMENT_COLORS,
    PROB_GO_COLOR, PROB_NOGO_COLOR, PROB_NEUTRAL_COLOR,
)


class HTMVisualizer:
    """Interactive 3D HTM Brain visualizer using PyVista.

    Modal keyboard system with four modes:
        NORMAL  (default)  Camera, UI controls
        SYNAPSE (S)        Synapse/connection visibility
        PICK    (P)        Click picking, multi-select, history
        CELL    (C)        Cell/segment colors + field toggles

    Press H for mode-aware shortcuts. ESC returns to NORMAL.
    """

    def __init__(self, brain, input_sequence: Iterable[dict[str, Any]] | None = None,
                 step_fn: Callable | None = None,
                 agent_step_fn: Callable[[int], dict[str, Any]] | None = None,
                 title: str = "HTM Visualizer"):
        self.brain = brain
        self.step_fn = step_fn
        self.agent_step_fn = agent_step_fn
        self.title = title

        # Support both lists and generators/iterables via unified iterator + cache
        self._input_iter = iter(input_sequence) if input_sequence else None
        self._input_cache: list[dict[str, Any]] = []

        self.timestep = 0
        self.history = History(max_size=1000)
        self.show_proximal = False
        self.learn = True
        self._external_mode = False  # True when start() is used instead of run()

        # Multi-selection list
        self._selections: list[dict] = []

        # Selection history
        self._selection_history: list[list[dict]] = []
        self._sel_hist_pos: int = -1

        # Saved synapse state for toggle on/off
        self._saved_synapse_state: dict | None = None

        # Legend and shortcuts visibility
        self._show_legend = False
        self._show_shortcuts = False
        self._show_speed_slider = True

        # Field visibility - set of field names that are hidden
        self.hidden_fields: set[str] = set()
        # Mapping of key -> field name for toggle shortcuts
        self._field_keys: dict[str, str] = {}

        # Metric tracking
        self.burst_history: list[int] = []
        self.error_history: list[float] = []

        # Modal keyboard system
        self.mode_manager = ModeManager()
        self.mode_manager.on_mode_change = self._on_mode_change

        # Renderers
        self.brain_renderer = BrainRenderer(brain)
        self.conn_renderer = ConnectionRenderer(self.brain_renderer)

        self.plotter = None

    def run(self):
        pv.global_theme.background = "black"
        pv.global_theme.font.color = "white"

        self.plotter = pv.Plotter(title=self.title, window_size=(1600, 900))
        self.plotter.set_background(pv.global_theme.background)
        self.plotter.enable_anti_aliasing("ssaa")
        
        # Disable stereo rendering to prevent VTK's '3' key from causing pink display
        self.plotter.iren.interactor.GetRenderWindow().SetStereoRender(False)
        self.plotter.iren.interactor.GetRenderWindow().SetStereoType(0)  # No stereo
        
        # Disable VTK's default 'p' key pick handler (which shows a red bounding box)
        style = self.plotter.iren.interactor.GetInteractorStyle()
        if style:
            style.PickingManagedOff()  # Disable VTK's built-in picking behavior

        self.brain_renderer.render_initial(self.plotter)

        self.playback = PlaybackController(
            step_callback=self._do_step,
            update_callback=self._update_display,
        )
        self.playback.step_back = self._step_back_history
        setup_key_bindings(self.plotter, self, self.mode_manager)

        # Selection via left-click: use iren click observer for ray-based picking
        self.plotter.iren.track_click_position(
            callback=self._on_click, side="left",
        )

        # self._add_title()
        self._add_controls_text()
        self._add_stats_overlay()
        self._add_selection_overlay()
        self._add_widgets()

        self._reset_camera()
        self._capture_snapshot({})
        self.plotter.show()

    def start(self):
        """Set up the plotter window for externally-driven stepping.

        Use this instead of run() when an external loop drives stepping.
        Call update(inputs) after each brain step — it will block until
        the user advances via RIGHT arrow or auto-play (SPACE).
        Call close() when done.
        """
        import time as _time
        self._time = _time

        pv.global_theme.background = "black"
        pv.global_theme.font.color = "white"

        self.plotter = pv.Plotter(title=self.title, window_size=(1600, 900))
        self.plotter.set_background(pv.global_theme.background)
        self.plotter.enable_anti_aliasing("ssaa")

        self.plotter.iren.interactor.GetRenderWindow().SetStereoRender(False)
        self.plotter.iren.interactor.GetRenderWindow().SetStereoType(0)

        style = self.plotter.iren.interactor.GetInteractorStyle()
        if style:
            style.PickingManagedOff()

        self.brain_renderer.render_initial(self.plotter)

        self._external_mode = True
        self._step_requested = False

        def _gate_step():
            self._step_requested = True

        self.playback = PlaybackController(
            step_callback=_gate_step,
            update_callback=lambda: None,
        )
        self.playback.step_back = self._step_back_history
        setup_key_bindings(self.plotter, self, self.mode_manager)

        self.plotter.iren.track_click_position(
            callback=self._on_click, side="left",
        )

        self._add_controls_text()
        self._add_stats_overlay()
        self._add_selection_overlay()
        self._add_widgets()

        self._reset_camera()
        self._capture_snapshot({})
        self.plotter.show(interactive_update=True)

        # Hook into brain.step() so update() fires automatically
        self.brain._post_step_hooks.append(self._on_brain_step)

    def _on_brain_step(self, brain, inputs: dict[str, Any], rewards: dict[str, Any], actions: dict[str, Any]):
        """Post-step hook — adapts the Brain callback signature to update()."""
        self.update(inputs)

    def update(self, inputs: dict[str, Any], predictions: dict | None = None):
        """Capture brain state, refresh display, then block until user advances.

        Call this from an external loop after each brain step.
        Blocks until the user presses RIGHT or auto-play timer fires.
        """
        if self.plotter is None:
            return
        self.timestep += 1
        self._track_metrics(inputs, predictions or {})
        self._capture_snapshot(inputs, predictions)
        self._update_display()

        # Wait for user to advance (RIGHT key or auto-play timer)
        self._step_requested = False
        while not self._step_requested:
            self.plotter.update()          # process VTK events (keys, timers)
            self._time.sleep(0.016)        # ~60 fps, avoid busy-spin

    def close(self):
        """Close the plotter window."""
        if self.plotter is not None:
            self.plotter.close()

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def _get_input_at(self, index: int) -> dict[str, Any] | None:
        """Get input at the given index, fetching from iterator if needed."""
        # Fetch items from iterator until we have enough cached
        while self._input_iter and len(self._input_cache) <= index:
            try:
                self._input_cache.append(next(self._input_iter))
            except StopIteration:
                self._input_iter = None
                break
        
        if not self._input_cache:
            return None
        
        # Return from cache, looping if index exceeds available items
        return self._input_cache[index % len(self._input_cache)]

    def _do_step(self):
        if self.agent_step_fn:
            # Agent callback handles the full brain+env loop (including brain.step)
            inputs = self.agent_step_fn(self.timestep)
            if inputs is None:
                return
            try:
                predictions = self.brain.prediction()
            except Exception:
                predictions = {}
        elif self.step_fn:
            inputs = self.step_fn(self.timestep)
            try:
                predictions = self.brain.prediction()
            except Exception:
                predictions = {}
            self.brain.step(inputs, learn=self.learn)
        else:
            inputs = self._get_input_at(self.timestep)
            if inputs is None:
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
            len(f.bursting_columns) for f in self.brain.all_column_fields.values()
        )
        self.burst_history.append(total_bursting)
        for name in self.brain.all_input_fields:
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
            for name in self.brain.all_column_fields:
                self.conn_renderer.render_proximal(self.plotter, name, active_only=False)

        # Selection highlights + synapse tracing (distal)
        self.brain_renderer.render_selection_highlights(self.plotter, self._selections)
        
        # Proximal synapses for selected columns
        self.conn_renderer.render_proximal_for_selection(self.plotter, self._selections)

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
                self.conn_renderer.render_proximal_for_selection(self.plotter, self._selections)
                self._update_stats_overlay()
                self.plotter.render()

    def _step_forward_history(self):
        if self.history.can_step_forward:
            self.history.step_forward()
            snap = self.history.current
            if snap:
                self.brain_renderer.update_from_snapshot(self.plotter, snap)
                self.brain_renderer.render_selection_highlights(self.plotter, self._selections)
                self.conn_renderer.render_proximal_for_selection(self.plotter, self._selections)
                self._update_stats_overlay()
                self.plotter.render()

    # ------------------------------------------------------------------
    # Selection (picking)
    # ------------------------------------------------------------------

    def _on_click(self, click_pos):
        """Handle left-click via ray-based picking for accurate selection."""
        if click_pos is None:
            return
        # Only pick in SELECT mode
        if self.mode_manager.current_mode != Mode.PICK:
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
            elif sel["type"] in ("segment", "apical_segment"):
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
            text = ""
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
            field_name = sel["field"]
            text = (
                f"CELL  {field_name} col={sel['col']} cell={sel['cell']}\n"
                f"  active:     {sel['active']}\n"
                f"  predictive: {sel['predictive']}\n"
                f"  winner:     {sel['winner']}\n"
            )
            # Go/NoGo depolarization
            if hasattr(cell, 'go_depolarized'):
                text += f"  go_depol:   {cell.go_depolarized}\n"
                text += f"  nogo_depol: {cell.nogo_depolarized}\n"

            # Column overlap
            col_obj = None
            field_obj = self.brain.all_column_fields.get(field_name)
            if field_obj and sel['col'] < len(field_obj.columns):
                col_obj = field_obj.columns[sel['col']]
                if hasattr(col_obj, 'overlap'):
                    text += f"  overlap:    {col_obj.overlap}\n"

            # TD value and eligibility trace (if in a ValueField)
            if hasattr(self.brain, '_value_fields') and field_name in self.brain._value_fields:
                vf = self.brain._value_fields[field_name]
                cell_flat_idx = sel['col'] * len(field_obj.columns[0].cells) + sel['cell']
                if cell_flat_idx < len(vf.values):
                    text += f"  td_value:   {vf.values[cell_flat_idx]:.4f}\n"
                    text += f"  trace:      {vf.traces[cell_flat_idx]:.4f}\n"

            # Distal segments
            text += f"  distal_seg: {len(cell.distal_segments)}\n"
            for si, seg in enumerate(cell.distal_segments):
                connected = sum(1 for s in seg.synapses if s.permanence >= 0.5)
                text += (
                    f"  d[{si}] syn={len(seg.synapses)} "
                    f"conn={connected} "
                    f"act={seg.active} lrn={seg.learning}\n"
                )

            # Apical segments
            apical_segs = getattr(cell, 'apical_segments', [])
            if apical_segs:
                go_segs = getattr(cell, 'go_segments', [])
                nogo_segs = getattr(cell, 'nogo_segments', [])
                text += f"  go_segs:    {len(go_segs)}\n"
                text += f"  nogo_segs:  {len(nogo_segs)}\n"
                for ai, aseg in enumerate(apical_segs):
                    sign_label = "Go" if aseg.sign > 0 else "NoGo"
                    connected = sum(1 for s in aseg.synapses if s.permanence >= 0.5)
                    text += (
                        f"  a[{ai}] {sign_label} syn={len(aseg.synapses)} "
                        f"conn={connected} "
                        f"score={aseg.score()} act={aseg.active}\n"
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
        elif sel["type"] == "apical_segment":
            aseg = sel["obj"]
            sign_label = "Go" if sel.get('sign', 1) > 0 else "NoGo"
            connected = sum(1 for s in aseg.synapses if s.permanence >= 0.5)
            perms = [s.permanence for s in aseg.synapses]
            text = (
                f"APICAL SEG ({sign_label})  {sel['field']} col={sel['col']} "
                f"cell={sel['cell']} seg={sel['seg']}\n"
                f"  synapses:   {sel['synapses']}\n"
                f"  connected:  {connected}\n"
                f"  score:      {sel['score']}\n"
                f"  active:     {sel['active']}\n"
                f"  learning:   {sel['learning']}\n"
            )
            if perms:
                text += (
                    f"  perm range: {min(perms):.3f} - {max(perms):.3f}\n"
                    f"  perm mean:  {sum(perms)/len(perms):.3f}\n"
                )
            return text
        elif sel["type"] == "input_cell":
            field_name = sel['field']
            text = (
                f"INPUT CELL  {field_name} idx={sel['index']}\n"
                f"  active:     {sel['active']}\n"
                f"  predictive: {sel['predictive']}\n"
            )
            # OutputField: show activation probability
            field_obj = self.brain.all_input_fields.get(field_name)
            if field_obj and hasattr(field_obj, 'activation_probabilities'):
                probs = field_obj.activation_probabilities()
                idx = sel['index']
                if idx < len(probs):
                    text += f"  act_prob:   {probs[idx]:.4f}\n"
            return text
        return ""

    # ------------------------------------------------------------------
    # Public control methods (called by key bindings)
    # ------------------------------------------------------------------

    def handle_escape(self):
        """ESC always exits to NORMAL, preserving selection."""
        if self.mode_manager.current_mode != Mode.NORMAL:
            self.mode_manager.exit_to_normal()

    def toggle_play(self):
        self.playback.toggle_play(self.plotter)

    def step_forward(self):
        # In start() mode, just release the gate — external loop does the stepping
        if self._external_mode:
            self._step_requested = True
            return
        # First try to step forward in history if we've stepped back
        if self.history.can_step_forward:
            self._step_forward_history()
        else:
            # Only run a new simulation step if at end of history
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

    def toggle_connected_proximal(self):
        """Toggle visibility of connected proximal synapses for selected columns."""
        self.conn_renderer.show_connected_proximal = not self.conn_renderer.show_connected_proximal
        self._update_display()

    def toggle_potential_proximal(self):
        """Toggle visibility of potential (not connected) proximal synapses for selected columns."""
        self.conn_renderer.show_potential_proximal = not self.conn_renderer.show_potential_proximal
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

    def toggle_go_apical(self):
        self.brain_renderer.show_go_apical = not self.brain_renderer.show_go_apical
        self._update_display()

    def toggle_nogo_apical(self):
        self.brain_renderer.show_nogo_apical = not self.brain_renderer.show_nogo_apical
        self._update_display()

    def toggle_output_synapses(self):
        self.brain_renderer.show_output_synapses = not self.brain_renderer.show_output_synapses
        self._update_display()

    def _get_synapse_state(self) -> dict:
        """Capture current synapse visibility flags."""
        return {
            "synapses": self.brain_renderer.show_synapses,
            "outgoing": self.brain_renderer.show_outgoing_synapses,
            "incoming": self.brain_renderer.show_incoming_synapses,
            "proximal": self.show_proximal,
            "connected": self.conn_renderer.show_connected_proximal,
            "potential": self.conn_renderer.show_potential_proximal,
            "go": self.brain_renderer.show_go_apical,
            "nogo": self.brain_renderer.show_nogo_apical,
            "output": self.brain_renderer.show_output_synapses,
        }

    def _set_synapse_state(self, state: dict):
        """Restore synapse visibility flags from a saved state."""
        self.brain_renderer.show_synapses = state["synapses"]
        self.brain_renderer.show_outgoing_synapses = state["outgoing"]
        self.brain_renderer.show_incoming_synapses = state["incoming"]
        self.show_proximal = state["proximal"]
        self.conn_renderer.show_connected_proximal = state["connected"]
        self.conn_renderer.show_potential_proximal = state["potential"]
        self.brain_renderer.show_go_apical = state["go"]
        self.brain_renderer.show_nogo_apical = state["nogo"]
        self.brain_renderer.show_output_synapses = state.get("output", True)

    def toggle_all_synapses(self):
        """Toggle between last active synapse state and all off."""
        all_off = not any(self._get_synapse_state().values())
        if all_off and self._saved_synapse_state:
            # Restore saved state
            self._set_synapse_state(self._saved_synapse_state)
            self._saved_synapse_state = None
        else:
            # Save current state, then turn all off
            self._saved_synapse_state = self._get_synapse_state()
            self._set_synapse_state({k: False for k in self._saved_synapse_state})
        self._update_display()

    def toggle_synapse_selection_only(self):
        """Toggle synapses to show only for selected elements."""
        self.brain_renderer.synapse_selection_only = not self.brain_renderer.synapse_selection_only
        self._update_display()

    def toggle_synapse_active_only(self):
        """Toggle synapses to show only for active segments."""
        self.brain_renderer.synapse_active_only = not self.brain_renderer.synapse_active_only
        self._update_display()

    def toggle_inactive(self):
        self.brain_renderer.hide_inactive = not self.brain_renderer.hide_inactive
        self._update_display()

    def toggle_state_color(self, state_name: str):
        """Toggle visibility of a specific cell state color."""
        if state_name in self.brain_renderer.hidden_states:
            self.brain_renderer.hidden_states.remove(state_name)
        else:
            self.brain_renderer.hidden_states.add(state_name)
        self._update_display()

    def toggle_segments(self):
        """Toggle global segment visibility."""
        self.brain_renderer.show_segments = not self.brain_renderer.show_segments
        self._update_display()

    def toggle_segment_state_color(self, state_name: str):
        """Toggle visibility of a specific segment state color."""
        if state_name in self.brain_renderer.hidden_segment_states:
            self.brain_renderer.hidden_segment_states.remove(state_name)
        else:
            self.brain_renderer.hidden_segment_states.add(state_name)
        self._update_display()

    def toggle_legend(self):
        self._show_legend = not self._show_legend
        self._update_legend()
        self.plotter.render()

    def toggle_shortcuts(self):
        self._show_shortcuts = not self._show_shortcuts
        self._update_shortcuts()
        self.plotter.render()

    def toggle_speed_slider(self):
        self._show_speed_slider = not self._show_speed_slider
        self._update_speed_slider()
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
        return list(self.brain.all_input_fields.keys()) + list(self.brain.all_column_fields.keys())

    def get_field_keys(self) -> dict[str, str]:
        """Return mapping of shortcut key -> field name."""
        return self._field_keys

    def set_field_keys(self, mapping: dict[str, str]):
        """Set the field key mapping."""
        self._field_keys = mapping

    # ------------------------------------------------------------------
    # UI elements
    # ------------------------------------------------------------------

    def _add_title(self):
        self.plotter.add_text(
            self.title, position="upper_left", font_size=10,
            color=color_to_float(TITLE_COLOR), name="title",
        )

    # Mode entries shown in the top hint bar: (key, label, Mode)
    _MODE_HINTS = [
        ("C", "Cell",    Mode.CELL),
        ("M", "Segment", Mode.SEGMENT),
        ("S", "Synapse", Mode.SYNAPSE),
        ("P", "Pick",    Mode.PICK),
    ]

    def _add_controls_text(self):
        # Dim base hint (always visible)
        self.plotter.add_text(
            "", position="upper_edge", font_size=9,
            color=(0.5, 0.5, 0.5), name="controls_hint", font="courier",
        )
        # Colored highlight overlay for the active mode
        self.plotter.add_text(
            "", position="upper_edge", font_size=9,
            color=(1.0, 1.0, 1.0), name="mode_highlight", font="courier",
        )
        self._shortcuts_actors = []
        self._update_controls_hint()

    def _update_controls_hint(self):
        """Rebuild the top hint bar, highlighting the active mode."""
        mode = self.mode_manager.current_mode
        dim = (0.5, 0.5, 0.5)

        parts = []
        highlight_text_parts = []
        for key, label, m in self._MODE_HINTS:
            entry = f"{key} {label}"
            parts.append(entry)
            if m == mode:
                highlight_text_parts.append(entry)
            else:
                # Invisible spacer of same width to keep alignment
                highlight_text_parts.append(" " * len(entry))

        suffix = "   H Shortcuts"
        base_text = "  ".join(parts) + suffix
        highlight_text = "  ".join(highlight_text_parts) + " " * len(suffix)

        if mode == Mode.NORMAL:
            # No mode active -- show all dim, no highlight
            self.plotter.add_text(
                base_text, position="upper_edge", font_size=9,
                color=dim, name="controls_hint", font="courier",
            )
            self.plotter.add_text(
                "", position="upper_edge", font_size=9,
                color=dim, name="mode_highlight", font="courier",
            )
        else:
            color = MODE_COLORS[mode]
            self.plotter.add_text(
                base_text, position="upper_edge", font_size=9,
                color=dim, name="controls_hint", font="courier",
            )
            self.plotter.add_text(
                highlight_text, position="upper_edge", font_size=9,
                color=color, name="mode_highlight", font="courier",
            )

    def _on_mode_change(self, mode: Mode):
        """Called when the mode changes -- update hint bar and shortcuts."""
        if self.plotter is None:
            return
        self._update_controls_hint()
        if self._show_shortcuts:
            self._update_shortcuts()
        self.plotter.render()

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
            for name in self.brain.all_input_fields:
                pred = snap.predictions.get(name)
                if pred is not None:
                    lines.append(f"  pred({name}): {pred:.4f}" if isinstance(pred, float) else f"  pred({name}): {pred}")
            lines.append("")
            for name in self.brain.all_column_fields:
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

        # ValueField TD learning stats
        if snap and snap.td_avg_error:
            lines.append("")
            for name in snap.td_avg_error:
                lines.append(f"[{name} TD]")
                lines.append(f"  Avg Error: {snap.td_avg_error[name]:.4f}")
                lines.append(f"  Avg Value: {snap.td_avg_value.get(name, 0.0):.4f}")

        # Apical segment counts
        if snap and snap.num_go_segments:
            for name in snap.num_go_segments:
                go_count = snap.num_go_segments.get(name, 0)
                nogo_count = snap.num_nogo_segments.get(name, 0)
                if go_count or nogo_count:
                    lines.append(f"  Go Segs:     {go_count}")
                    lines.append(f"  NoGo Segs:   {nogo_count}")

        # OutputField decoded action and confidence
        if snap and snap.output_decoded:
            for name, decoded in snap.output_decoded.items():
                lines.append(f"\n[{name} Output]")
                val = decoded.get("value") if isinstance(decoded, dict) else None
                conf = decoded.get("confidence") if isinstance(decoded, dict) else None
                lines.append(f"  Action:     {val}")
                if conf is not None:
                    lines.append(f"  Confidence: {conf:.4f}")

        if self.burst_history:
            recent = self.burst_history[-20:]
            lines.append(f"\nBurst Avg(20): {sum(recent)/len(recent):.1f}")
        if self.error_history:
            recent = self.error_history[-20:]
            lines.append(f"Error Avg(20): {sum(recent)/len(recent):.4f}")

        lines.append(f"\nHistory: {self.history._position + 1}/{len(self.history)}")

        # Compact synapse status
        syn_flags = []
        if self.brain_renderer.show_synapses:
            syn_flags.append("Dist")
        if self.show_proximal:
            syn_flags.append("Prox")
        if self.conn_renderer.show_connected_proximal:
            syn_flags.append("Conn")
        if self.conn_renderer.show_potential_proximal:
            syn_flags.append("Pot")
        if self.brain_renderer.show_outgoing_synapses:
            syn_flags.append("Out")
        if self.brain_renderer.show_incoming_synapses:
            syn_flags.append("In")
        if self.brain_renderer.show_go_apical:
            syn_flags.append("Go")
        if self.brain_renderer.show_nogo_apical:
            syn_flags.append("NoGo")
        lines.append(f"Synapses: {' '.join(syn_flags) if syn_flags else 'all OFF'}")
        if self.brain_renderer.hide_inactive:
            lines.append("Hide Inactive: ON")

        # Compact hidden colors
        hidden = self.brain_renderer.hidden_states | self.brain_renderer.hidden_segment_states
        if hidden:
            lines.append(f"Hidden colors: {', '.join(sorted(hidden))}")

        # Field visibility
        if self.hidden_fields:
            lines.append(f"Hidden fields: {', '.join(sorted(self.hidden_fields))}")

        return "\n".join(lines)

    def _add_widgets(self):
        self._speed_slider = None
        self._update_speed_slider()

    def _update_speed_slider(self):
        # Remove existing slider if present
        if self._speed_slider is not None:
            self.plotter.clear_slider_widgets()
            self._speed_slider = None

        if not self._show_speed_slider:
            return

        def speed_callback(value):
            self.playback.speed_ms = int(value)

        self._speed_slider = self.plotter.add_slider_widget(
            speed_callback, rng=[1, 2000], value=self.playback.speed_ms,
            title="Speed (ms)", pointa=(0.8, 0.05), pointb=(0.98, 0.05),
            style="modern", color=color_to_float(TEXT_COLOR),
            tube_width=0.005, slider_width=0.01,
            title_height=0.023,
        )

    # Map internal key names to display labels for shortcuts HUD
    _KEY_DISPLAY = {
        "space": "SPACE", "Escape": "ESC",
        "Left": "<- ->", "Right": None,  # Right is shown combined with Left
        "bracketleft": "[", "bracketright": "]",
    }

    @staticmethod
    def _display_key(key: str) -> str | None:
        """Convert an internal key name to a display string, or None to skip."""
        table = HTMVisualizer._KEY_DISPLAY
        if key in table:
            return table[key]
        return key.upper() if len(key) == 1 else key

    @staticmethod
    @staticmethod
    def _parse_mnemonic(desc: str) -> tuple[str, int]:
        """Parse a description with a _ mnemonic marker.

        The _ character marks the next letter as the mnemonic.
        Returns (clean_desc, mnemonic_index) where mnemonic_index is
        the position in the cleaned string, or -1 if no marker found.
        """
        idx = desc.find("_")
        if idx < 0 or idx + 1 >= len(desc):
            return desc, -1
        clean = desc[:idx] + desc[idx + 1:]
        return clean, idx

    def _format_shortcut_block(
        self, label: str, mode: Mode,
        sections: dict[str, list[tuple[str, str]]],
        global_bindings: list[tuple[str, str]],
    ) -> tuple[str, str]:
        """Build two fixed-width text blocks for the shortcuts overlay.

        Returns (base_text, highlight_mask) where:
        - base_text: full text rendered in dim color
        - highlight_mask: spaces everywhere except mnemonic letters, rendered bright

        Both are padded to the same width for VTK alignment.
        """
        # First pass: collect all (display_key, raw_desc) pairs to measure key_width
        # raw_desc may contain & mnemonic markers
        all_pairs: list[tuple[str, str]] = []

        if mode == Mode.PICK:
            all_pairs.append(("Click", "Pick"))
            all_pairs.append(("Shift+Click", "Multi-pick"))

        for sec_name, bindings in sections.items():
            for key, desc in bindings:
                dk = self._display_key(key)
                if dk is not None:
                    all_pairs.append((dk, desc))

        for key, desc in global_bindings:
            dk = self._display_key(key)
            if dk is not None:
                all_pairs.append((dk, desc))

        # Measure widths using cleaned descriptions (& stripped)
        key_width = max((len(k) for k, _ in all_pairs), default=5)
        key_width = max(key_width, 5)

        def row(dk: str, raw_desc: str) -> tuple[str, str]:
            """Return (base_line, highlight_line) for one shortcut row."""
            clean_desc, mi = self._parse_mnemonic(raw_desc)
            base = f"{dk:>{key_width}}  {clean_desc}"
            if mi >= 0:
                # Offset into full line: key_width + 2 spaces + index in desc
                abs_idx = key_width + 2 + mi
                highlight = " " * abs_idx + clean_desc[mi] + " " * (len(base) - abs_idx - 1)
            else:
                highlight = " " * len(base)
            return base, highlight

        # Second pass: build lines with section headers
        base_lines: list[str] = []
        hl_lines: list[str] = []

        def add(base: str, hl: str = ""):
            base_lines.append(base)
            hl_lines.append(hl if hl else " " * len(base))

        add("")
        add(f"  [{label} MODE]")

        if mode == Mode.PICK:
            b, h = row("Click", "Pick")
            add(b, h)
            b, h = row("Shift+Click", "Multi-pick")
            add(b, h)

        for sec_name, bindings in sections.items():
            if sec_name:
                add("")
                add(f"{sec_name:>{key_width + 2}}")
            for key, desc in bindings:
                dk = self._display_key(key)
                if dk is not None:
                    b, h = row(dk, desc)
                    add(b, h)

        # Global section
        add("")
        add(f"{'Global':>{key_width + 2}}")
        for key, desc in global_bindings:
            dk = self._display_key(key)
            if dk is not None:
                b, h = row(dk, desc)
                add(b, h)

        width = max(len(line) for line in base_lines)
        base_text = "\n".join(line.ljust(width) for line in base_lines)
        hl_text = "\n".join(line.ljust(width) for line in hl_lines)
        return base_text, hl_text

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

        mode = self.mode_manager.current_mode
        label = MODE_LABELS[mode]
        sections = self.mode_manager.get_bindings_by_section()
        global_bindings = self.mode_manager.get_global_bindings_for_display()

        base_text, hl_text = self._format_shortcut_block(label, mode, sections, global_bindings)

        # Base layer: dim text with all characters
        dim_color = (0.55, 0.55, 0.55)
        actor_base = self.plotter.add_text(
            base_text,
            position="upper_right",
            font_size=10,
            color=dim_color,
            name="shortcuts_box",
            font="courier",
        )
        actor_base.GetTextProperty().SetBackgroundColor(0.0, 0.0, 0.0)
        actor_base.GetTextProperty().SetBackgroundOpacity(0)
        self._shortcuts_actors.append(actor_base)

        # Highlight layer: bright mnemonic letters only (rest spaces)
        actor_hl = self.plotter.add_text(
            hl_text,
            position="upper_right",
            font_size=10,
            color=(1.0, 1.0, 1.0),
            name="shortcuts_highlight",
            font="courier",
        )
        actor_hl.GetTextProperty().SetBackgroundOpacity(0)
        self._shortcuts_actors.append(actor_hl)

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
            ("Active Cell", color_to_float(COLORS["active"])),
            ("Predictive Cell", color_to_float(COLORS["predictive"])),
            ("Bursting Cell", color_to_float(COLORS["bursting"])),
            ("Winner Cell", color_to_float(COLORS["winner"])),
            ("Correct Prediction", color_to_float(COLORS["correct_prediction"])),
            ("Go Depolarized", color_to_float(COLORS["go_depolarized"])),
            ("NoGo Depolarized", color_to_float(COLORS["nogo_depolarized"])),
            ("Inactive Cell", color_to_float(COLORS["inactive"])),
        ]

        segment_entries = [
            ("Active Segment", color_to_float(SEGMENT_COLORS["active"])),
            ("Learning Segment", color_to_float(SEGMENT_COLORS["learning"])),
            ("Matching Segment", color_to_float(SEGMENT_COLORS["matching"])),
            ("Inactive Segment", color_to_float(SEGMENT_COLORS["inactive"])),
        ]

        apical_entries = [
            ("Go Apical Segment", color_to_float(APICAL_SEGMENT_COLORS["go_active"])),
            ("NoGo Apical Segment", color_to_float(APICAL_SEGMENT_COLORS["nogo_active"])),
            ("Apical Learning Segment", color_to_float(APICAL_SEGMENT_COLORS["learning"])),
        ]

        prob_entries = [
            ("Probability Go (above baseline)", color_to_float(PROB_GO_COLOR)),
            ("Probability Neutral (at baseline)", color_to_float(PROB_NEUTRAL_COLOR)),
            ("Probability NoGo (below baseline)", color_to_float(PROB_NOGO_COLOR)),
        ]

        all_entries = cell_entries + segment_entries + apical_entries + prob_entries

        legend_actor = self.plotter.add_legend(
            labels=all_entries,
            bcolor=(0.0, 0.0, 0.0),
            border=False,
            size=(0.15, 0.35),
            name="legend",
            font_family="courier",
        )
        self._legend_actors.append(legend_actor)

    def _reset_camera(self):
        self.plotter.camera_position = "xz"
        self.plotter.camera.azimuth = 20
        self.plotter.camera.elevation = 20
        self.plotter.reset_camera()
        self.plotter.camera.zoom(0.8)
