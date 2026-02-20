"""Launch the HTM 3D Brain Visualizer with a sine wave model."""

import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from core.HTM import ColumnField, InputField
from core.brain import Brain
from src.encoder_layer.rdse import RDSEParameters
from visualizer import HTMVisualizer


def build_sine_model():
    """Build a simple HTM model for sine wave prediction."""
    params = RDSEParameters(
        size=128,
        sparsity=0.02,
        resolution=0.1,
        seed=42,
    )
    value_field = InputField(encoder_params=params)

    l1 = ColumnField(
        input_fields=[value_field],
        non_spatial=True,
        num_columns=128,
        cells_per_column=8,
    )

    brain = Brain({
        "value": value_field,
        "L1": l1,
    })
    return brain


def generate_sine_sequence(n_steps=500, period=50):
    """Generate sine wave input sequence."""
    return [
        {"value": math.sin(2 * math.pi * i / period) * 5 + 5}
        for i in range(n_steps)
    ]


def main():
    brain = build_sine_model()
    sequence = generate_sine_sequence()

    viz = HTMVisualizer(
        brain,
        input_sequence=sequence,
        title="HTM Visualizer",
    )
    viz.run()


if __name__ == "__main__":
    main()
