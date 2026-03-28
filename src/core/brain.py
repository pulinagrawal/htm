"""Field manager for simplified HTM input handling.

Encapsulates InputFields and ColumnFields to provide a unified API
for encoding inputs and computing temporal memory in a single step.
"""

from typing import Any, Callable

from core.HTM import ColumnField, InputField, Field, OutputField
from core.sungur import ValueField

# Hook signature: (brain, inputs, rewards, behaviors) -> None
PostStepHook = Callable[["Brain", dict[str, Any], dict[str, Any], dict[str, Any]], None]


class Brain:
    """Manages HTM input fields and column fields with a unified API.

    Allows binding named inputs to InputFields and processing all inputs
    with a single `step()` call instead of manually calling encode/compute.

    Example:
        manager = FieldManager()
        manager.add_input_field("consumption", consumption_field)
        manager.add_input_field("date", date_field)
        manager.set_column_field(column_field)

        # Training loop - single call replaces manual encode/compute
        for row in data:
            result = manager.step({
                "consumption": row["value"],
                "date": row["timestamp"],
            })
    """

    def __init__(self, fields: dict[str, Field]) -> None:
        self._output_fields: dict[str, OutputField] = {k:v for k,v in fields.items() if isinstance(v, OutputField)}
        self._input_fields: dict[str, InputField] = {k:v for k,v in fields.items() if isinstance(v, InputField)
                                                      and not isinstance(v, OutputField)}
        self._value_fields: dict[str, ValueField] = {k:v for k,v in fields.items() if isinstance(v, ValueField)}
        self._column_fields: dict[str, ColumnField] = {k:v for k,v in fields.items() if isinstance(v, ColumnField)
                                                        and not isinstance(v, ValueField)}
        self.fields = fields
        self._post_step_hooks: list[PostStepHook] = []

        # Combined dicts for rendering — all fields that share the same visual shape
        self.all_column_fields: dict[str, ColumnField] = {
            k: v for k, v in fields.items() if isinstance(v, ColumnField)
        }
        self.all_input_fields: dict[str, InputField] = {
            k: v for k, v in fields.items()
            if isinstance(v, InputField) and not isinstance(v, ColumnField)
        }
    
    def __getitem__(self, name: str) -> Field:
        return self.fields[name]

    def __getattr__(self, name: str) -> Field:
        if name in self.fields:
            return self.fields[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def step(
        self,
        inputs: dict[str, Any],
        learn: bool = True,
        reward: float | None = None,
    ):
        """Process one timestep: encode all inputs and compute column field.

        Args:
            inputs: Dict mapping field names to input values.
            learn: Whether to enable learning during this step.
            reward: External reward signal. If None, reward is computed internally.
        """
        self.encode_only(inputs)
        self.compute_only(learn=learn)
        if 'reward' in inputs:
            reward = inputs['reward']
        self.estimate_value(reward)
        self.activate_apical_segments()
        actions = self.generate_behavior()
        for hook in self._post_step_hooks:
            hook(self, inputs, {'reward': reward}, actions)
        return actions

    def estimate_value(self, reward: float | None = None) -> None:
        for name, field in self._value_fields.items():
            if reward is None:
                reward = self.compute_intrinsic_reward()
            field.update_values(reward)

    def activate_apical_segments(self) -> None:
        """Activate apical segments in Go/NoGo fields based on current column states."""
        for field in self._column_fields.values():
            field.apical_compute()
    
    def generate_behavior(self) -> dict[str, Any]:
        """Compute output fields and return decoded values.

        Returns:
            Dict mapping output field names to their decoded values,
            suitable for feeding directly to an environment.
        """
        for field in self._output_fields.values():
            field.compute()
        behavior = {}
        for name, field in self._output_fields.items():
            result = field.decode()
            behavior[name] = result["value"]
        return behavior

    def encode_only(self, inputs: dict[str, Any]) -> None:
        """Encode inputs without computing (useful for getting predictions first).
    
        Args:
            inputs: Dict mapping field names to input values.
        """
        for name in self._input_fields:
            if name in inputs:
                self._input_fields[name].encode(inputs[name])

    def compute_only(self, learn: bool = True) -> None:
        """Compute column field without encoding (inputs already encoded).

        Args:
            learn: Whether to enable learning during this step.
        """
        # TODO: No distinction between column and value fields right now
        for field in self._column_fields.values():
            field.compute(learn=learn)
        for field in self._value_fields.values():
            field.compute(learn=learn)
    
    def compute_intrinsic_reward(self) -> float:
        """Compute an intrinsic reward signal"""
        raise NotImplementedError("Needs Implementation")

    def print_stats(self) -> None:
        """Print statistics from the column field."""
        for name, column_field in self._column_fields.items():
            print(f"Statistics for ColumnField '{name}':")
            column_field.print_stats()

    def reset(self) -> None:
        """Clear all states in the column field."""
        for field in self.fields:
            self.fields[field].reset()

    def prediction(self) -> tuple[Any, ...]:
        """Get the current prediction for a specific input field.

        Returns:
            Result from decoder (typically value, confidence tuple).

        Raises:
            KeyError: If field_name doesn't match a registered field.
            ValueError: If ColumnField is not set.
        """
        predictions = {}
        for input_name in self._input_fields:
            input_field = self._input_fields[input_name]
            if hasattr(input_field.encoder, 'decode'):
                predictions[input_name], predictions[input_name+'.conf'] = input_field.decode('predictive')
        
        return predictions
