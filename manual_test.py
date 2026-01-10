#!/usr/bin/env python3
"""Hyperparameter search to satisfy the sine-wave bursting regression test."""
from __future__ import annotations

from tqdm import tqdm
import random

import numpy as np

from HTM import ColumnField, InputField
from rdse import RandomDistributedScalarEncoder as RDSE, RDSEParameters


def test_sine_wave_bursting_columns_converge():
        """Test ColumnField bursts converge to zero on a learned sine-driven sequence."""
        config = {
            "num_columns": 512,
            "cells_per_column": 16,
            "activation_threshold": 3,
            "learning_threshold": 5,
            "resolution": 0.001,
            "cycle_length": 64,
            "rdse_seed": 5,
            "total_steps": 1000,
        }

        np.random.seed(42)
        random.seed(42)
        params = RDSEParameters(
            size=config["num_columns"],
            sparsity=0.02,
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

        for step in tqdm(range(config["total_steps"])):
            value = sine_cycle[step % config["cycle_length"]]
            input_field.encode(value)
            column_field.compute()
            burst_counts.append(len(column_field.bursting_columns))

        print("Burst counts over time:", burst_counts)

        column_field.print_stats()

        evaluation_bursts = []
        errors = []

        for value in sine_cycle:
            input_fields = column_field.get_prediction()
            prediction, confidence = input_field.decode(input_fields[0], 'predictive')
            errors.append(abs(value - prediction)**2)
            input_field.encode(value)
            column_field.compute(learn=False)
            evaluation_bursts.append(len(column_field.bursting_columns))


        # mean abs error of predictions
        mae = sum(errors) / len(errors)
        print("Mean Absolute Error of predictions:", mae)
        print("Evaluation bursting columns:", evaluation_bursts)

if __name__ == "__main__":
    test_sine_wave_bursting_columns_converge()

""""
Stats 

learning =.5*num_synapses
ColumnField statistics:
  Columns: 512 | Cells: 8192 | Segments: 114 | Synapses: 16731
  +------------------------+--------------------+----------+----------+
  | Metric                 |   Mean ± Std      |      Min |      Max |
  +------------------------+--------------------+----------+----------+
  | Segments per cell     |     0.01 ± 0.33    |        0 |       16 |
  | Synapses per segment  |   146.76 ± 134.99  |       16 |      496 |
  | Permanence            |    0.145 ± 0.197   |    0.000 |    1.000 |
  +------------------------+--------------------+----------+----------+
  Connected synapses (>= 0.5): 492 (2.9% of all synapses)

learning = 5
ColumnField statistics:
  Columns: 512 | Cells: 8192 | Segments: 51 | Synapses: 11263
  +------------------------+--------------------+----------+----------+
  | Metric                 |   Mean ± Std      |      Min |      Max |
  +------------------------+--------------------+----------+----------+
  | Segments per cell     |     0.01 ± 0.10    |        0 |        2 |
  | Synapses per segment  |   220.84 ± 191.64  |        0 |      559 |
  | Permanence            |    0.047 ± 0.199   |    0.000 |    1.000 |
  +------------------------+--------------------+----------+----------+
  Connected synapses (>= 0.5): 524 (4.7% of all synapses)

learning = .1*num_synapses
ColumnField statistics:
  Columns: 512 | Cells: 8192 | Segments: 38 | Synapses: 11327
  +------------------------+--------------------+----------+----------+
  | Metric                 |   Mean ± Std       |      Min |      Max |
  +------------------------+--------------------+----------+----------+
  | Segments per cell      |     0.00 ± 0.08    |        0 |        4 |
  | Synapses per segment   |   298.08 ± 160.36  |       16 |      559 |
  | Permanence             |    0.048 ± 0.199   |    0.000 |    1.000 |
  +------------------------+--------------------+----------+----------+
  Connected synapses (>= 0.5): 524 (4.6% of all synapses)

activation threshold = .005*max_synapses
max_new_synapse_count=.1*field
ColumnField statistics:
  Columns: 512 | Cells: 8192 | Segments: 38 | Synapses: 11102
  +------------------------+--------------------+----------+----------+
  | Metric                 |   Mean ± Std      |      Min |      Max |
  +------------------------+--------------------+----------+----------+
  | Segments per cell     |     0.00 ± 0.08    |        0 |        4 |
  | Synapses per segment  |   292.16 ± 157.54  |       16 |      559 |
  | Permanence            |    0.049 ± 0.200   |    0.000 |    1.000 |
  +------------------------+--------------------+----------+----------+
  Connected synapses (>= 0.5): 524 (4.7% of all synapses)
"""