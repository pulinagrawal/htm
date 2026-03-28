"""Modal keyboard system for the HTM Visualizer.

Modes group related shortcuts so each mode has a small, focused keyset.
Mode-switch keys (S, P, C) only work from NORMAL mode -- no cross-mode jumping.
"""

from enum import Enum, auto
from typing import Callable


class Mode(Enum):
    NORMAL = auto()
    SYNAPSE = auto()
    PICK = auto()
    CELL = auto()
    SEGMENT = auto()


MODE_COLORS: dict[Mode, tuple[float, float, float]] = {
    Mode.NORMAL:  (1.0, 1.0, 1.0),
    Mode.SYNAPSE: (0.0, 0.9, 0.9),
    Mode.PICK:    (1.0, 0.9, 0.2),
    Mode.CELL:    (0.9, 0.3, 0.9),
    Mode.SEGMENT: (0.3, 0.9, 0.5),
}

MODE_LABELS: dict[Mode, str] = {
    Mode.NORMAL:  "NORMAL",
    Mode.SYNAPSE: "SYNAPSE",
    Mode.PICK:    "PICK",
    Mode.CELL:    "CELL",
    Mode.SEGMENT: "SEGMENT",
}


class ModeManager:
    """Dispatches key events based on the current mode.

    Keys are registered per-mode or as globals. On each keypress, globals are
    checked first, then the current mode's bindings.
    """

    def __init__(self):
        self.current_mode: Mode = Mode.NORMAL
        # mode -> {key -> (callback, description, section)}
        self._bindings: dict[Mode, dict[str, tuple[Callable, str, str]]] = {
            m: {} for m in Mode
        }
        # {key -> (callback, description)}
        self._global_bindings: dict[str, tuple[Callable, str]] = {}
        self.on_mode_change: Callable[[Mode], None] | None = None

    def register(self, mode: Mode, key: str, callback: Callable,
                 description: str, section: str = ""):
        """Register a key binding for a specific mode."""
        self._bindings[mode][key] = (callback, description, section)

    def register_global(self, key: str, callback: Callable, description: str):
        """Register a key binding that works in all modes."""
        self._global_bindings[key] = (callback, description)

    def enter_mode(self, target: Mode):
        """Enter a mode. Only works from NORMAL."""
        if self.current_mode != Mode.NORMAL:
            return
        if target == Mode.NORMAL:
            return
        self.current_mode = target
        if self.on_mode_change:
            self.on_mode_change(target)

    def exit_to_normal(self):
        """Return to NORMAL mode."""
        if self.current_mode == Mode.NORMAL:
            return
        self.current_mode = Mode.NORMAL
        if self.on_mode_change:
            self.on_mode_change(Mode.NORMAL)

    def dispatch(self, key: str):
        """Dispatch a key event. Globals first, then current mode."""
        if key in self._global_bindings:
            self._global_bindings[key][0]()
            return
        mode_bindings = self._bindings.get(self.current_mode, {})
        if key in mode_bindings:
            mode_bindings[key][0]()

    def get_bindings_by_section(self) -> dict[str, list[tuple[str, str]]]:
        """Return current mode's bindings grouped by section.

        Returns {section_name: [(key, description), ...]} sorted by key
        within each section. Sections are ordered: '' first, then alphabetical.
        """
        mode_bindings = self._bindings.get(self.current_mode, {})
        sections: dict[str, list[tuple[str, str]]] = {}
        for k, (_, desc, section) in sorted(mode_bindings.items()):
            sections.setdefault(section, []).append((k, desc))
        # Return with default section first
        result = {}
        if "" in sections:
            result[""] = sections.pop("")
        for sec in sorted(sections):
            result[sec] = sections[sec]
        return result

    def get_global_bindings_for_display(self) -> list[tuple[str, str]]:
        """Return (key, description) pairs for global bindings."""
        return [(k, desc) for k, (_, desc) in sorted(self._global_bindings.items())]

    def all_registered_keys(self) -> set[str]:
        """Return all keys registered across all modes and globals."""
        keys = set(self._global_bindings.keys())
        for mode_bindings in self._bindings.values():
            keys.update(mode_bindings.keys())
        return keys
