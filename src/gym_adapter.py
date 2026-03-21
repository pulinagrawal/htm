"""Wrapper that gives Brain a Gymnasium-compatible interface.

Wraps Brain so callers can use a simple step(obs, reward) → action
loop without knowing about Brain's dict-based encode/decode internals.
"""

from __future__ import annotations

from typing import Any, Callable

from core.brain import Brain


class GymBrain:
    """Brain wrapper with a Gymnasium-friendly step interface.

    Translates between Brain's dict I/O and the flat obs/action
    values that a Gymnasium loop expects.

    Args:
        brain: The underlying HTM Brain.
        obs_to_inputs: Converts an env observation to a Brain input dict.
        behavior_to_action: Converts Brain.step() behavior dict to an env action.
    """

    def __init__(
        self,
        brain: Brain,
        obs_to_inputs: Callable[[Any], dict[str, Any]],
        behavior_to_action: Callable[[dict[str, Any]], Any],
    ) -> None:
        self.brain = brain
        self.obs_to_inputs = obs_to_inputs
        self.behavior_to_action = behavior_to_action

    def step(self, obs: Any, reward: float = 0.0, learn: bool = True) -> Any:
        """Process one observation and return an action.

        Args:
            obs: Raw environment observation.
            reward: Reward from the previous transition.
            learn: Whether to enable learning.

        Returns:
            An action compatible with the target environment's action space.
        """
        inputs = self.obs_to_inputs(obs)
        behavior = self.brain.step(inputs=inputs, learn=learn, reward=reward)
        return self.behavior_to_action(behavior)

    def reset(self) -> None:
        """Reset the underlying Brain state."""
        self.brain.reset()
