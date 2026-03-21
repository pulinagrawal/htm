"""Base class for all encoders.

This module provides the abstract base class for encoder implementations,
ported from NuPIC's C++ codebase. An encoder converts a value to a sparse
distributed representation (SDR).

All encoder implementations must satisfy three critical properties:

1. **Semantic similarity**: Similar inputs should have high overlap. Overlap
   decreases smoothly as inputs become less similar. Dissimilar inputs have
   very low overlap so that the output representations are not easily confused.

2. **Stability**: The representation for an input does not change during the
   lifetime of the encoder.

3. **Sparsity**: The output SDR should have a similar sparsity for all inputs
   and have enough active bits to handle noise and subsampling.

Reference:
    https://arxiv.org/pdf/1602.05925.pdf - HTM whitepaper
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, override, runtime_checkable

T = TypeVar("T")


class BaseEncoder(ABC, Generic[T]):
    """Abstract base class for all encoder implementations.

    Encoders convert input values into Sparse Distributed Representations (SDRs).
    Subclasses must implement the `encode` method and can optionally override
    the `reset` method.

    Args:
        size: Total number of bits in the output SDR. If None, defaults to 0.
    """

    def __init__(self, size: int | None = None):
        self._size: int = size if size is not None else 0
        self._parameters: ParameterMarker | None = None

    @property
    def size(self) -> int:
        """Get the total number of bits in the output SDR.

        Returns:
            Total size of the SDR.
        """
        assert self._size >= 0, "size must be a non-negative integer"
        return self._size

    @size.setter
    def size(self, value: int) -> None:
        """Set the total number of bits in the output SDR.

        Args:
            value: New size for the SDR. Must be positive.

        Raises:
            ValueError: If value is non-positive.
        """
        if value < 0:
            raise ValueError("size must be a non-negative integer")
        elif value == 0:
            raise ValueError("size must be greater than zero")
        self._size = value

    def reset(self) -> None:
        """Reset the encoder to its initial state.

        Clears dimensions, size, and any buffered data. Subclasses
        can override this method to add additional reset logic.
        """
        self._dimensions = []
        self._size = 0
        self.__buffered_data = None
        self.__buffer_bounds = None

    @abstractmethod
    def encode(self, input_value: T) -> list[int]:
        """Encode an input value into a sparse distributed representation.

        Subclasses must implement this method to define how input values are
        transformed into SDRs.

        Args:
            input_value: The value to encode (type determined by generic parameter T).

        Returns:
            Binary list of 0s and 1s representing the SDR.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement the encoding method")

    @abstractmethod
    def decode(self, input_sdr: list[int]) -> Any:
        """Decodes the input sdr into a value and confidence."""
        raise NotImplementedError("Subclasses must implement the decoding method")


@runtime_checkable
class ParameterMarker(Protocol):
    """Parent dataclass for encoder parameter configurations.

    Provides base configuration fields common to all encoder types. Subclasses
    should override encoder_class to reference their specific encoder type and
    add encoder-specific parameters.

    Attributes:
        encoder_class: The encoder class associated with these parameters.
        size: Total size of the output SDR in bits.
    """

    encoder_class = BaseEncoder
    """Reference to the encoder class associated with these parameters. Subclasses should override this to point to their specific encoder type."""

    size: int = 2048
    """Total size of the output SDR in bits."""