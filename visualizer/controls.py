"""Playback controls and interaction handling."""

from typing import Callable


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


def setup_key_bindings(plotter, app):
    """Configure keyboard shortcuts."""
    plotter.add_key_event("space", lambda: app.toggle_play())
    plotter.add_key_event("Right", lambda: app.step_forward())
    plotter.add_key_event("Left", lambda: app.step_back())
    plotter.add_key_event("r", lambda: app.reset_view())
    plotter.add_key_event("p", lambda: app.toggle_proximal())
    plotter.add_key_event("s", lambda: app.toggle_synapses())
    plotter.add_key_event("o", lambda: app.toggle_outgoing_synapses())
    plotter.add_key_event("i", lambda: app.toggle_incoming_synapses())
    plotter.add_key_event("a", lambda: app.toggle_inactive())
    plotter.add_key_event("l", lambda: app.toggle_legend())
    plotter.add_key_event("h", lambda: app.toggle_shortcuts())
    plotter.add_key_event("Escape", lambda: app.clear_selection())
    plotter.add_key_event("bracketleft", lambda: app.selection_back())
    plotter.add_key_event("bracketright", lambda: app.selection_forward())

    # Setup field visibility toggles using first available letter from field name
    _setup_field_key_bindings(plotter, app)


# Keys reserved by other shortcuts or PyVista
RESERVED_KEYS = {
    "r", "p", "s", "o", "i", "l", "h", "a",  # Our shortcuts
    "q", "e", "f", "v", "w", "c",            # Common PyVista keys
    "3",                                       # PyVista stereo mode
}


def _setup_field_key_bindings(plotter, app):
    """Assign letter shortcuts to fields based on their names."""
    field_names = app.get_field_names()
    used_keys: set[str] = set(RESERVED_KEYS)
    field_keys: dict[str, str] = {}  # key -> field_name

    for field_name in field_names:
        # Find first available letter from the field name
        assigned_key = None
        for char in field_name.lower():
            if char.isalpha() and char not in used_keys:
                assigned_key = char
                used_keys.add(char)
                break

        if assigned_key:
            field_keys[assigned_key] = field_name

            # Create closure properly
            def make_toggle(fname):
                return lambda: app.toggle_field(fname)

            plotter.add_key_event(assigned_key, make_toggle(field_name))

    # Store the mapping in the app for display purposes
    app.set_field_keys(field_keys)
