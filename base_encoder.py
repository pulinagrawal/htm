"""Base class for all encoders -- from NuPic Numenta Cpp ported to python.
/**
 * Base class for all encoders.
 * An encoder converts a value to a sparse distributed representation.
 *
 * Subclasses must implement method encode and Serializable interface.
 * Subclasses can optionally implement method reset.
 *
 * There are several critical properties which all encoders must have:
 *
 * 1) Semantic similarity:  Similar inputs should have high overlap.  Overlap
 * decreases smoothly as inputs become less similar.  Dissimilar inputs have
 * very low overlap so that the output representations are not easily confused.
 *
 * 2) Stability:  The representation for an input does not change during the
 * lifetime of the encoder.
 *
 * 3) Sparsity: The output SDR should have a similar sparsity for all inputs and
 * have enough active bits to handle noise and subsampling.
 *
 * Reference: https://arxiv.org/pdf/1602.05925.pdf
 */

"""

from abc import ABC, abstractmethod
from math import prod
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class BaseEncoder(ABC, Generic[T]):
    """Base class for all encoders"""

    def __init__(self, dimensions: list[int] | None = None, size: int | None = None):
        """Initializes the BaseEncoder with given dimensions."""

        self._dimensions: list[int] = dimensions if dimensions is not None else []
        self._size: int = size if size is not None else prod(int(dim) for dim in self._dimensions)

    @property
    def dimensions(self) -> list[int]:
        return self._dimensions

    @property
    def size(self) -> int:
        return self._size

    def reset(self):
        """Resets the encoder to its initial state if applicable."""

        self._dimensions = []
        self._size = 0


    @abstractmethod
    def encode(self, input_value: T) -> Any:
        """Encodes the input value into the provided output SDR by reference."""
        raise NotImplementedError("Subclasses must implement this method")