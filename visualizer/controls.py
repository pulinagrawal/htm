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

    def toggle_play(self, plotter):
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
        def on_timer(obj, event):
            if self.playing:
                self.step_forward()

        iren = plotter.iren
        iren.add_timer_event(on_timer)
        self._timer_id = iren.create_timer(self.speed_ms, repeating=True)

    def _stop_timer(self, plotter):
        if self._timer_id is not None:
            plotter.iren.destroy_timer(self._timer_id)
            self._timer_id = None


def setup_key_bindings(plotter, app):
    """Configure keyboard shortcuts."""
    plotter.add_key_event("space", lambda: app.toggle_play())
    plotter.add_key_event("Right", lambda: app.step_forward())
    plotter.add_key_event("Left", lambda: app.step_back())
    plotter.add_key_event("r", lambda: app.reset_view())
    plotter.add_key_event("p", lambda: app.toggle_proximal())
    plotter.add_key_event("s", lambda: app.toggle_synapses())
    plotter.add_key_event("Escape", lambda: app.clear_selection())
