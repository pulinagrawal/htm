"""Playback controls and interaction handling."""

from typing import Callable

from .mode_manager import Mode, ModeManager


class PlaybackController:
    """Manages playback state: play/pause, step, speed."""

    def __init__(self, step_callback: Callable, update_callback: Callable):
        self.step_callback = step_callback
        self.update_callback = update_callback
        self.playing = False
        self.speed_ms = 500
        self._timer_id = None
        self._observer_id = None
        self._plotter = None

    def toggle_play(self, plotter):
        self._plotter = plotter
        self.playing = not self.playing
        if self.playing:
            self._start_timer(plotter)
        else:
            self._stop_timer(plotter)

    def step_forward(self):
        self.step_callback()
        self.update_callback()

    def step_back(self):
        pass  # Overridden by app

    def set_speed(self, speed_ms: int):
        self.speed_ms = max(50, speed_ms)

    def _start_timer(self, plotter):
        # Clean up any existing timer first
        self._stop_timer(plotter)

        def on_timer(obj, event):
            if self.playing:
                self.step_forward()

        vtk_iren = plotter.iren.interactor
        self._observer_id = vtk_iren.AddObserver("TimerEvent", on_timer)
        self._timer_id = vtk_iren.CreateRepeatingTimer(self.speed_ms)

    def _stop_timer(self, plotter):
        vtk_iren = plotter.iren.interactor
        if self._timer_id is not None:
            vtk_iren.DestroyTimer(self._timer_id)
            self._timer_id = None
        if self._observer_id is not None:
            vtk_iren.RemoveObserver(self._observer_id)
            self._observer_id = None


# Keys reserved by PyVista internals (not available for any mode)
_PYVISTA_RESERVED = {"q", "e", "f", "w", "3"}


def setup_key_bindings(plotter, app, mode_manager: ModeManager):
    """Configure keyboard shortcuts via the modal ModeManager."""

    # --- Global keys (work in all modes) ---
    mode_manager.register_global("space", lambda: app.toggle_play(), "Play/Pause")
    mode_manager.register_global("Right", lambda: app.step_forward(), "Step forward")
    mode_manager.register_global("Left", lambda: app.step_back(), "Step back")
    mode_manager.register_global("h", lambda: app.toggle_shortcuts(), "Help")
    mode_manager.register_global("Escape", lambda: app.handle_escape(), "Back / Clear")

    # --- NORMAL mode ---
    # Use _ before the mnemonic letter to mark it for highlighting
    # Modes (ordered: Cells, Synapses, Pick)
    mode_manager.register(Mode.NORMAL, "c", lambda: mode_manager.enter_mode(Mode.CELL), "_Cell mode toggle", section='Modes')
    mode_manager.register(Mode.NORMAL, "m", lambda: mode_manager.enter_mode(Mode.SEGMENT), "Seg_ment mode toggle", section='Modes')
    mode_manager.register(Mode.NORMAL, "s", lambda: mode_manager.enter_mode(Mode.SYNAPSE), "_Synapse mode toggle", section='Modes')
    mode_manager.register(Mode.NORMAL, "p", lambda: mode_manager.enter_mode(Mode.PICK), "_Pick mode toggle", section='Modes')

    # Visibility
    mode_manager.register(Mode.NORMAL, "i", lambda: app.toggle_inactive(), "_Inactive cell visibility", section="Visibility")
    mode_manager.register(Mode.NORMAL, "v", lambda: app.toggle_all_synapses(), "_Visibile Synapses", section="Visibility")
    mode_manager.register(Mode.NORMAL, "o", lambda: app.toggle_synapse_selection_only(), "_Only selected visible", section="Visibility")
    mode_manager.register(Mode.NORMAL, "a", lambda: app.toggle_synapse_active_only(), "_Active elements only", section="Visibility")

    # Shortcuts
    mode_manager.register(Mode.NORMAL, "r", lambda: app.reset_view(), "_Reset camera", section="Shortcuts")
    mode_manager.register(Mode.NORMAL, "l", lambda: app.toggle_legend(), "_Legend", section="Shortcuts")
    mode_manager.register(Mode.NORMAL, "t", lambda: app.toggle_speed_slider(), "_Toggle Speed slider", section="Shortcuts")

    # --- SYNAPSE mode (_ marks mnemonic letter) ---
    mode_manager.register(Mode.SYNAPSE, "s", lambda: mode_manager.exit_to_normal(), "_Synapse mode toggle")
    mode_manager.register(Mode.SYNAPSE, "d", lambda: app.toggle_synapses(), "_Distal Visibility", section="Distal")
    mode_manager.register(Mode.SYNAPSE, "o", lambda: app.toggle_outgoing_synapses(), "_Outgoing from cell", section="Distal")
    mode_manager.register(Mode.SYNAPSE, "i", lambda: app.toggle_incoming_synapses(), "_Incoming to segment", section="Distal")
    mode_manager.register(Mode.SYNAPSE, "a", lambda: app.toggle_output_synapses(), "_Action synapses", section="Distal")
    mode_manager.register(Mode.SYNAPSE, "p", lambda: app.toggle_proximal(), "_Proximal Visibility", section="Proximal")
    mode_manager.register(Mode.SYNAPSE, "c", lambda: app.toggle_connected_proximal(), "_Connected proximal", section="Proximal")
    mode_manager.register(Mode.SYNAPSE, "u", lambda: app.toggle_potential_proximal(), "_Unconnected proximal", section="Proximal")
    mode_manager.register(Mode.SYNAPSE, "g", lambda: app.toggle_go_apical(), "_Go apical synapses", section="Apical")
    mode_manager.register(Mode.SYNAPSE, "n", lambda: app.toggle_nogo_apical(), "_NoGo apical synapses", section="Apical")

    # --- PICK mode ---
    mode_manager.register(Mode.PICK, "p", lambda: mode_manager.exit_to_normal(), "_Pick mode toggle")
    mode_manager.register(Mode.PICK, "x", lambda: app.clear_selection(), "Clear selection")
    mode_manager.register(Mode.PICK, "bracketleft", lambda: app.selection_back(), "Selection history back")
    mode_manager.register(Mode.PICK, "bracketright", lambda: app.selection_forward(), "Selection history forward")

    # --- CELL mode (_ marks mnemonic letter) ---
    mode_manager.register(Mode.CELL, "c", lambda: mode_manager.exit_to_normal(), "_Cell mode toggle")
    mode_manager.register(Mode.CELL, "a", lambda: app.toggle_state_color("active"), "_Active cells")
    mode_manager.register(Mode.CELL, "p", lambda: app.toggle_state_color("predictive"), "_Predictive cells")
    mode_manager.register(Mode.CELL, "b", lambda: app.toggle_state_color("bursting"), "_Bursting cells")
    mode_manager.register(Mode.CELL, "w", lambda: app.toggle_state_color("winner"), "_Winner cells")
    mode_manager.register(Mode.CELL, "y", lambda: app.toggle_state_color("correct_prediction"), "Correctl_y predicted")
    mode_manager.register(Mode.CELL, "g", lambda: app.toggle_state_color("go_depolarized"), "_Go depolarized")
    mode_manager.register(Mode.CELL, "n", lambda: app.toggle_state_color("nogo_depolarized"), "_NoGo depolarized")
    # Field visibility toggles (CELL mode, dynamic)
    _setup_field_key_bindings(mode_manager, app)

    # --- SEGMENT mode ---
    mode_manager.register(Mode.SEGMENT, "m", lambda: mode_manager.exit_to_normal(), "Seg_ment mode toggle")
    mode_manager.register(Mode.SEGMENT, "v", lambda: app.toggle_segments(), "Segment _visibility")
    mode_manager.register(Mode.SEGMENT, "a", lambda: app.toggle_segment_state_color("active"), "_Active segments")
    mode_manager.register(Mode.SEGMENT, "l", lambda: app.toggle_segment_state_color("learning"), "_Learning segments")
    mode_manager.register(Mode.SEGMENT, "s", lambda: app.toggle_segment_state_color("matching"), "Matching _segments")

    # --- Wire keys directly via a VTK observer with high priority ---
    # This fires BEFORE VTK's interactor style processes the key.
    # We clear the key symbol after dispatching so VTK's OnChar finds
    # nothing to act on (prevents built-in t=timer, s=surface, etc.).
    managed_keys = mode_manager.all_registered_keys()
    iren = plotter.iren.interactor

    def on_key_press(obj, event):
        key = obj.GetKeySym()
        if key and key in managed_keys:
            mode_manager.dispatch(key)
            obj.SetKeySym("")
            obj.SetKeyCode("\0")

    iren.AddObserver("KeyPressEvent", on_key_press, 100.0)


# Keys reserved in CELL mode (color toggles + mode switches + PyVista)
_CELL_RESERVED = {
    "a", "p", "b", "w", "y", "g", "n", "c",  # Cell color toggles + mode toggle
    "h",  # Global
} | _PYVISTA_RESERVED


def _setup_field_key_bindings(mode_manager: ModeManager, app):
    """Assign letter shortcuts to fields, registered under CELL mode."""
    field_names = app.get_field_names()
    used_keys: set[str] = set(_CELL_RESERVED)
    field_keys: dict[str, str] = {}  # key -> field_name

    for field_name in field_names:
        assigned_key = None
        mnemonic_idx = -1
        # First try: letters from the field name itself
        for i, char in enumerate(field_name):
            if char.isalpha() and char.lower() not in used_keys:
                assigned_key = char.lower()
                mnemonic_idx = i
                break
        # Fallback: first available letter in the alphabet
        if assigned_key is None:
            for char in "abcdefghijklmnopqrstuvwxyz":
                if char not in used_keys:
                    assigned_key = char
                    break

        if assigned_key:
            used_keys.add(assigned_key)
            field_keys[assigned_key] = field_name

            # Insert _ marker at the mnemonic position
            if mnemonic_idx >= 0:
                label = field_name[:mnemonic_idx] + "_" + field_name[mnemonic_idx:]
            else:
                # Fallback key not in name — append marker
                label = f"{field_name} (_{assigned_key.upper()})"

            def make_toggle(fname):
                return lambda: app.toggle_field(fname)

            mode_manager.register(
                Mode.CELL, assigned_key,
                make_toggle(field_name),
                label,
                section="Fields",
            )

    app.set_field_keys(field_keys)
