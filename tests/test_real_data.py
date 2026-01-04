import random
import unittest

import numpy as np

from building_blocks import InputField, ColumnField
from rdse import RDSEParameters

class TestRealData(unittest.TestCase):

    def test_sine_wave_bursting_columns_converge(self):
        """Test ColumnField bursts converge to zero on a learned sine-driven sequence."""
        np.random.seed(42)
        random.seed(42)
        num_columns = 256
        cells_per_column = 6
        params = RDSEParameters(
            size=num_columns,
            active_bits=16,
            sparsity=0.0,
            radius=0.25,
            resolution=0.0,
            category=False,
            seed=11,
        )
        input_field = InputField(size=num_columns, rdse_params=params)
        column_field = ColumnField(
            input_fields=[input_field],
            non_spatial=True,
            n_columns=num_columns,
            n_cells_per_column=cells_per_column,
        )

        cycle_length = 64
        sine_cycle = np.sin(np.linspace(0, 2 * np.pi, cycle_length, endpoint=False))
        burst_counts = []
        total_steps = 1000

        for step in range(total_steps):
            value = sine_cycle[step % cycle_length]
            input_field.encode(value)
            column_field.compute()
            burst_counts.append(sum(column.bursting for column in column_field.columns))

        self.assertGreater(
            max(burst_counts[:10]),
            0,
            "Column field should burst before learning the sine-driven sequence.",
        )
        print("Burst counts over time:", burst_counts)
        self.assertEqual(
            burst_counts[-1],
            0,
            "Bursting columns should converge to zero after 1000 sine inputs.",
        )

        evaluation_bursts = []

        for value in sine_cycle:
            input_field.encode(value)
            column_field.compute()
            evaluation_bursts.append(sum(column.bursting for column in column_field.columns))

        self.assertTrue(
            all(count == 0 for count in evaluation_bursts),
            f"Expected no bursting once the sine sequence is mastered, got {evaluation_bursts}",
        )
