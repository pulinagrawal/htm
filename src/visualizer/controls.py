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
    mode_manager.register_global("h", lambda: app.toggle_shortcuts(), "Shortcuts")
    mode_manager.register_global("Escape", lambda: app.handle_escape(), "Back / Clear")

    # --- NORMAL mode ---
    mode_manager.register(Mode.NORMAL, "v", lambda: mode_manager.enter_mode(Mode.SYNAPSE), "Synapse mode")
    mode_manager.register(Mode.NORMAL, "m", lambda: mode_manager.enter_mode(Mode.SELECT), "Select mode")
    mode_manager.register(Mode.NORMAL, "c", lambda: mode_manager.enter_mode(Mode.COLOR), "Color mode")
    mode_manager.register(Mode.NORMAL, "r", lambda: app.reset_view(), "Reset camera")
    mode_manager.register(Mode.NORMAL, "a", lambda: app.toggle_inactive(), "Hide inactive")
    mode_manager.register(Mode.NORMAL, "l", lambda: app.toggle_legend(), "Legend")
    mode_manager.register(Mode.NORMAL, "t", lambda: app.toggle_speed_slider(), "Speed slider")

    # --- SYNAPSE mode (mnemonic letters) ---
    mode_manager.register(Mode.SYNAPSE, "d", lambda: app.toggle_synapses(), "Distal(all)")
    mode_manager.register(Mode.SYNAPSE, "p", lambda: app.toggle_proximal(), "Proximal(all)")
    mode_manager.register(Mode.SYNAPSE, "c", lambda: app.toggle_connected_proximal(), "Prox Connected")
    mode_manager.register(Mode.SYNAPSE, "u", lambda: app.toggle_potential_proximal(), "Prox Potential")
    mode_manager.register(Mode.SYNAPSE, "o", lambda: app.toggle_outgoing_synapses(), "Outgoing(cell)")
    mode_manager.register(Mode.SYNAPSE, "i", lambda: app.toggle_incoming_synapses(), "Incoming(seg)")
    mode_manager.register(Mode.SYNAPSE, "g", lambda: app.toggle_go_apical(), "Go apical")
    mode_manager.register(Mode.SYNAPSE, "n", lambda: app.toggle_nogo_apical(), "NoGo apical")

    # --- SELECT mode ---
    mode_manager.register(Mode.SELECT, "x", lambda: app.clear_selection(), "Clear select")
    mode_manager.register(Mode.SELECT, "bracketleft", lambda: app.selection_back(), "Hist back")
    mode_manager.register(Mode.SELECT, "bracketright", lambda: app.selection_forward(), "Hist forward")

    # --- COLOR mode (mnemonic letters) ---
    mode_manager.register(Mode.COLOR, "a", lambda: app.toggle_state_color("active"), "Active")
    mode_manager.register(Mode.COLOR, "p", lambda: app.toggle_state_color("predictive"), "Predictive")
    mode_manager.register(Mode.COLOR, "b", lambda: app.toggle_state_color("bursting"), "Bursting")
    mode_manager.register(Mode.COLOR, "w", lambda: app.toggle_state_color("winner"), "Winner")
    mode_manager.register(Mode.COLOR, "k", lambda: app.toggle_state_color("correct_prediction"), "Correct Pred")
    mode_manager.register(Mode.COLOR, "g", lambda: app.toggle_state_color("go_depolarized"), "Go Depol")
    mode_manager.register(Mode.COLOR, "n", lambda: app.toggle_state_color("nogo_depolarized"), "NoGo Depol")
    mode_manager.register(Mode.COLOR, "s", lambda: app.toggle_segment_state_color("active"), "Seg Active")
    mode_manager.register(Mode.COLOR, "l", lambda: app.toggle_segment_state_color("learning"), "Seg Learning")
    mode_manager.register(Mode.COLOR, "m", lambda: app.toggle_segment_state_color("matching"), "Seg Matching")

    # --- Field visibility toggles (NORMAL mode, dynamic) ---
    _setup_field_key_bindings(mode_manager, app)

    # --- Wire all registered keys to PyVista via a single dispatcher per key ---
    for key in mode_manager.all_registered_keys():
        def make_handler(k):
            def handler():
                mode_manager.dispatch(k)
            return handler
        plotter.add_key_event(key, make_handler(key))


# Keys reserved in NORMAL mode (our shortcuts + mode switches + PyVista)
NORMAL_RESERVED = {
    "r", "a", "l", "t", "h", "v", "m", "c",  # NORMAL mode keys
} | _PYVISTA_RESERVED


def _setup_field_key_bindings(mode_manager: ModeManager, app):
    """Assign letter shortcuts to fields, registered under NORMAL mode."""
    field_names = app.get_field_names()
    used_keys: set[str] = set(NORMAL_RESERVED)
    field_keys: dict[str, str] = {}  # key -> field_name

    for field_name in field_names:
        assigned_key = None
        for char in field_name.lower():
            if char.isalpha() and char not in used_keys:
                assigned_key = char
                used_keys.add(char)
                break

        if assigned_key:
            field_keys[assigned_key] = field_name

            def make_toggle(fname):
                return lambda: app.toggle_field(fname)

            mode_manager.register(
                Mode.NORMAL, assigned_key,
                make_toggle(field_name),
                field_name,
                section="Fields",
            )

    app.set_field_keys(field_keys)
