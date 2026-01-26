"""Field manager for simplified HTM input handling.

Encapsulates InputFields and ColumnFields to provide a unified API
for encoding inputs and computing temporal memory in a single step.
"""

from typing import Any

from src.HTM import ColumnField, InputField, Field, OutputField


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
        self._input_fields: dict[str, InputField] = {k:v for k,v in fields.items() if isinstance(v, InputField)}
        self._output_fields: dict[str, OutputField] = {k:v for k,v in fields.items() if isinstance(v, OutputField)}
        self._column_fields: dict[str, ColumnField] = {k:v for k,v in fields.items() if isinstance(v, ColumnField)}
        self.fields = fields
    
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
    ):
        """Process one timestep: encode all inputs and compute column field.

        Args:
            inputs: Dict mapping field names to input values.
            learn: Whether to enable learning during this step.
        """
        self.encode_only(inputs)
        self.compute_only(learn=learn)
        return {name: field.decode() for name, field in self._output_fields.items()}

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

    def encode_only(self, inputs: dict[str, Any]) -> None:
        """Encode inputs without computing (useful for getting predictions first).

        Args:
            inputs: Dict mapping field names to input values.
        """
        for name, value in inputs.items():
            if name not in self._input_fields:
                raise KeyError(f"Unknown input field: '{name}'")
            self._input_fields[name].encode(value)

    def compute_only(self, learn: bool = True) -> None:
        """Compute column field without encoding (inputs already encoded).

        Args:
            learn: Whether to enable learning during this step.
        """
        # Compute temporal memory
        for column_field in self._column_fields.values():
            column_field.compute(learn=learn)

    def print_stats(self) -> None:
        """Print statistics from the column field."""
        for name, column_field in self._column_fields.items():
            print(f"Statistics for ColumnField '{name}':")
            column_field.print_stats()

    def reset(self) -> None:
        """Clear all states in the column field."""
        for field in self.fields:
            field.reset()
