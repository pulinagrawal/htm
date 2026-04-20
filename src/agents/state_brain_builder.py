from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gymnasium import spaces
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.HTM import ColumnField, InputField
from core.brain import Brain
from encoder_layer.category import CategoryParametersNew
from encoder_layer.rdse import RDSEParameters

CATEGORY_ENCODER_SIZE = 100
CATEGORY_ACTIVE_BITS = 10
RDSE_ENCODER_SIZE = 100
RDSE_SPARSITY = 0.02
RDSE_RESOLUTION = 0.1
RDSE_SEED = 5
CELLS_PER_COLUMN = 8


@dataclass(frozen=True)
class StateFieldSpec:
    name: str
    encoder_kind: str
    category_list: list[Any] | None = None
    allow_new_categories: bool = False


def python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def is_integer_like(value: Any) -> bool:
    value = python_scalar(value)
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def is_float_like(value: Any) -> bool:
    value = python_scalar(value)
    return isinstance(value, (float, np.floating))


def flatten_state(state: Any) -> list[Any]:
    if isinstance(state, np.ndarray):
        return [python_scalar(value) for value in state.reshape(-1)]
    if isinstance(state, (list, tuple)):
        return [python_scalar(value) for value in np.asarray(state, dtype=object).reshape(-1)]
    return [python_scalar(state)]


def bounded_integer_categories(low: Any, high: Any) -> list[int] | None:
    low = python_scalar(low)
    high = python_scalar(high)
    if not is_integer_like(low) or not is_integer_like(high):
        return None
    span = int(high) - int(low)
    if span < 0 or span > 128:
        return None
    return list(range(int(low), int(high) + 1))


def infer_state_specs(
    observation_space: spaces.Space[Any] | None = None,
    state_sample: Any | None = None,
) -> list[StateFieldSpec]:
    if isinstance(observation_space, spaces.Discrete):
        return [
            StateFieldSpec(
                name="state",
                encoder_kind="category",
                category_list=list(range(observation_space.n)),
            )
        ]

    if isinstance(observation_space, spaces.Box):
        flat_low = np.asarray(observation_space.low).reshape(-1)
        flat_high = np.asarray(observation_space.high).reshape(-1)
        is_integer_box = np.issubdtype(observation_space.dtype, np.integer)
        specs: list[StateFieldSpec] = []

        for index, (low, high) in enumerate(zip(flat_low, flat_high, strict=False)):
            name = "state" if flat_low.size == 1 else f"state_{index}"
            if is_integer_box:
                categories = bounded_integer_categories(low, high)
                specs.append(
                    StateFieldSpec(
                        name=name,
                        encoder_kind="category",
                        category_list=categories,
                        allow_new_categories=categories is None,
                    )
                )
            else:
                specs.append(StateFieldSpec(name=name, encoder_kind="rdse"))
        return specs

    if state_sample is None:
        raise ValueError("build_brain requires an observation_space or state_sample")

    flat_state = flatten_state(state_sample)
    specs: list[StateFieldSpec] = []
    for index, value in enumerate(flat_state):
        name = "state" if len(flat_state) == 1 else f"state_{index}"
        if is_integer_like(value):
            specs.append(
                StateFieldSpec(
                    name=name,
                    encoder_kind="category",
                    allow_new_categories=True,
                )
            )
        elif is_float_like(value):
            specs.append(StateFieldSpec(name=name, encoder_kind="rdse"))
        else:
            raise TypeError(f"Unsupported state value type for '{name}': {type(value)!r}")
    return specs


def build_input_field(spec: StateFieldSpec) -> InputField:
    if spec.encoder_kind == "category":
        encoder_params = CategoryParametersNew(
            size=CATEGORY_ENCODER_SIZE,
            active_bits_per_category=CATEGORY_ACTIVE_BITS,
            category_list=spec.category_list or [],
            new_categories_allowed=spec.allow_new_categories,
        )
        field = InputField(encoder_params=encoder_params)
        for category in spec.category_list or []:
            field.encoder.encode(category)
        return field

    if spec.encoder_kind == "rdse":
        encoder_params = RDSEParameters(
            size=RDSE_ENCODER_SIZE,
            sparsity=RDSE_SPARSITY,
            resolution=RDSE_RESOLUTION,
            category=False,
            seed=RDSE_SEED,
        )
        return InputField(encoder_params=encoder_params)

    raise ValueError(f"Unknown encoder kind: {spec.encoder_kind}")


def state_to_inputs(state: Any, input_names: list[str] | None = None) -> dict[str, Any]:
    flat_state = flatten_state(state)
    if input_names is None:
        input_names = ["state"] if len(flat_state) == 1 else [f"state_{index}" for index in range(len(flat_state))]
    if len(input_names) != len(flat_state):
        raise ValueError("State shape does not match brain input field layout")
    return {name: python_scalar(value) for name, value in zip(input_names, flat_state, strict=False)}


def build_typed_brain(
    observation_space: spaces.Space[Any] | None = None,
    state_sample: Any | None = None,
) -> Brain:
    state_specs = infer_state_specs(observation_space=observation_space, state_sample=state_sample)
    input_fields = {spec.name: build_input_field(spec) for spec in state_specs}

    column_field = ColumnField(
        input_fields=list(input_fields.values()),
        non_spatial=True,
        cells_per_column=CELLS_PER_COLUMN,
    )

    brain_fields: dict[str, Any] = dict(input_fields)
    brain_fields["column"] = column_field
    brain = Brain(brain_fields)
    brain.state_input_names = list(input_fields.keys())
    brain.state_field_specs = state_specs
    return brain