# HTM ColumnField Architecture

This repository contains a compact HTM-inspired implementation focused on ColumnField-based spatial/temporal memory and encoder-driven input fields. It is designed for time-series experimentation with sparse distributed representations (SDRs).

## Overview

The core engine lives in [src/HTM.py](src/HTM.py) and provides:

- **Field**: This is an important abstraction pervasive this codebase. A field is a collection of cells. This is useful for passing around the states and representations encoded in the states of the cells in the field.
- **InputField / OutputField**: Encapsulate encoders and expose active/predictive cells.
- **ColumnField**: Combines spatial pooling and temporal memory in a single field.  
       - **Note**: ColumnField has a non-spatial mode for pure temporal memory.
- **Cell / Segment / Synapse**: Distal learning primitives used by ColumnField.
- **Brain** (in [src/brain.py](src/brain.py)): A small orchestration layer that wires multiple fields and runs a single `step()` across them.

Encoders live under [src/encoder_layer](src/encoder_layer), including a Random Distributed Scalar Encoder (RDSE) and a rich DateEncoder for calendar features.

## Features

- **ColumnField compute loop** that performs overlap, inhibition, temporal memory, and prediction propagation.
- **RDSE and DateEncoder** for scalar and time-of-day style inputs.
- **Multi-input fields** (e.g., consumption + date) with prediction support.
- **Experiment scripts** in [src](src) for real datasets (Hot Gym) and synthetic sine-wave evaluation.

## Project layout

- [src/HTM.py](src/HTM.py): Core HTM data structures and `ColumnField` implementation.
- [src/brain.py](src/brain.py): `Brain` wrapper to encode/compute with one call.
- [src/encoder_layer](src/encoder_layer): RDSE, DateEncoder, and base encoder utilities.
- [src/hot_gym_model.py](src/hot_gym_model.py): Training/evaluation pipeline for the Hot Gym dataset.
- [src/manual_test.py](src/manual_test.py): Sine-wave regression evaluator.
- [data/rec-center-hourly.csv](data/rec-center-hourly.csv): Example dataset used by scripts/tests.
- [tests](tests): pytest suite for encoders and real-data behavior.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Hot Gym evaluation:

```bash
python src/hot_gym_model.py
```

Run the sine-wave evaluator:

```bash
python src/manual_test.py
```

## Core usage example

```python
from src.HTM import ColumnField, InputField
from src.encoder_layer.rdse import RDSEParameters
from src.brain import Brain

rdse_params = RDSEParameters(
    size=1024,
    sparsity=0.02,
    resolution=0.1,
    category=False,
    seed=5,
)

input_field = InputField(encoder_params=rdse_params)
column_field = ColumnField(
    input_fields=[input_field],
    non_spatial=True,
    num_columns=1024,
    cells_per_column=16,
)

brain = Brain({"signal": input_field, "column_field": column_field})

value = 1.23
brain.step({"signal": value})
prediction = brain.prediction()["signal"]
```

## Configuration knobs

Key constants live in [src/HTM.py](src/HTM.py):

- `CONNECTED_PERM`: permanence threshold for connected synapses
- `DESIRED_LOCAL_SPARSITY`: target sparse activation in the spatial pooler
- `INITIAL_PERMANENCE`: permanence assigned to new synapses
- `PERMANENCE_INC` / `PERMANENCE_DEC`: learning step size
- `ACTIVATION_THRESHOLD_PCT` / `LEARNING_THRESHOLD_PCT`: segment thresholds
- `MAX_SYNAPSE_PCT`: cap on synapses per segment

These can be edited directly or parameterized in downstream experiments.

## Requirements

- Python 3.13+
- NumPy
- pytest

## Testing

Run the full test suite with:

```bash
pytest
```

## Contributing

This is a research-oriented codebase. Keep changes minimal, readable, and well-documented, and prefer pytest for tests.
