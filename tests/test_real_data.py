import random
import unittest

import numpy as np

from building_blocks import InputField, ColumnField
from rdse import RDSEParameters

class TestRealData(unittest.TestCase):

    def test_sine_wave_bursting_columns_converge(self):
        """Test ColumnField bursts converge to zero on a learned sine-driven sequence."""
        config = {
            "num_columns": 512,
            "cells_per_column": 16,
            "activation_threshold": 3,
            "learning_threshold": 5,
            "active_bits": 16,
            "resolution": 0.1,
            "cycle_length": 64,
            "rdse_seed": 5,
            "total_steps": 1000,
        }

        np.random.seed(42)
        random.seed(42)
        params = RDSEParameters(
            size=config["num_columns"],
            active_bits=config["active_bits"],
            sparsity=0.0,
            radius=0.0,
            resolution=config["resolution"],
            category=False,
            seed=config["rdse_seed"],
        )
        input_field = InputField(size=config["num_columns"], rdse_params=params)
        column_field = ColumnField(
            input_fields=[input_field],
            non_spatial=True,
            num_columns=config["num_columns"],
            cells_per_column=config["cells_per_column"],
        )

        sine_cycle = np.sin(
            np.linspace(0, 2 * np.pi, config["cycle_length"], endpoint=False)
        )
        burst_counts = []

        for step in range(config["total_steps"]):
            value = sine_cycle[step % config["cycle_length"]]
            input_field.encode(value)
            column_field.compute()
            burst_counts.append(len(column_field.bursting_columns))

        self.assertGreater(
            max(burst_counts[:10]),
            0,
            "Column field should burst before learning the sine-driven sequence.",
        )
        print("Burst counts over time:", burst_counts)

        column_field.print_stats()
        self.assertEqual(
            burst_counts[-1],
            0,
            "Bursting columns should converge to zero after 1000 sine inputs.",
        )

        evaluation_bursts = []

        for value in sine_cycle:
            input_field.encode(value)
            column_field.compute(learn=False)
            evaluation_bursts.append(len(column_field.bursting_columns))

        burst_tolerance_pct = .05
        self.assertTrue(
            sum(count for count in evaluation_bursts)<= burst_tolerance_pct * len(evaluation_bursts),
            f"Expected tolerant bursting once the sine sequence is mastered, got {evaluation_bursts}",
        )