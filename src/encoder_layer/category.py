"""Category encoder for discrete, unrelated categorical values.

This module provides an encoder for discrete categories represented by strings
that have no inherent relationship or ordering. Each category is encoded to
a distinct, non-overlapping sparse distributed representation.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, override

from encoder_layer.base_encoder import BaseEncoder, ParameterMarker
from log.log import get_logger, logger


class CategoryEncoderNew(BaseEncoder[str]):
    """Encoder for discrete categorical values with no semantic relationships.

    This encoder converts string categories into sparse distributed representations
    where each category maps to a distinct, non-overlapping pattern. Unlike scalar
    encoders, there is no similarity between different categories' representations.

    Encodings are pre-computed at initialization using random non-overlapping
    bit allocations, guaranteeing that no two categories share any active bits.

    Args:
        parameters: Configuration specifying categories and encoder settings.
    """

    def __init__(self, parameters: CategoryParametersNew):
        self._parameters = copy.deepcopy(parameters)
        self._category_list = list(self._parameters.category_list)
        self._num_categories = len(self._category_list)
        self._new_categories_allowed = self._parameters.new_categories_allowed

        self.size = self._parameters.size

        if self._parameters.sparsity != 0:
            self._active_bits = int(round(self.size * self._parameters.sparsity))
        else:
            self._active_bits = self._parameters.active_bits_per_category

        if self._active_bits <= 0:
            raise ValueError("active_bits must be positive (set sparsity or active_bits_per_category)")

        required_bits = (self._num_categories + 1) * self._active_bits
        if required_bits > self.size:
            raise ValueError(
                f"Not enough bits ({self.size}) for {self._num_categories + 1} categories "
                f"with {self._active_bits} active bits each (need {required_bits})"
            )

        self._max_categories = self.size // self._active_bits - 1  # minus 1 for unknown slot

        self.logger = get_logger(self)
        self._encoding_cache: dict[str, list[int]] = {}

        self._category_encodings: dict[str | None, list[int]] = {}
        self._precompute_encodings(self._parameters.seed)

        super().__init__(self._size)

    def _precompute_encodings(self, seed: int) -> None:
        """Pre-compute guaranteed non-overlapping encodings for all known categories."""
        rng = random.Random(seed)
        self._shuffled_bits = list(range(self.size))
        rng.shuffle(self._shuffled_bits)
        self._next_slot = 0

        # Slot 0: unknown/unrecognized categories
        self._category_encodings[None] = self._allocate_slot()

        # Slots 1..N: known categories
        for category in self._category_list:
            self._category_encodings[category] = self._allocate_slot()

    def _allocate_slot(self) -> list[int]:
        """Allocate the next non-overlapping bit block and return the encoding."""
        start = self._next_slot * self._active_bits
        positions = set(self._shuffled_bits[start:start + self._active_bits])
        self._next_slot += 1
        return [1 if i in positions else 0 for i in range(self.size)]

    @property
    def active_bits(self) -> int:
        """Number of active bits per category."""
        return self._active_bits

    @override
    def encode(self, input_value: Any) -> list[int]:
        """Encode a category string into a sparse distributed representation.

        When new_categories_allowed=True and the category is unseen, a new
        unique encoding is allocated if capacity remains. Otherwise falls back
        to the unknown encoding.

        Args:
            input_value: Category string to encode.

        Returns:
            Binary list of 0s and 1s representing the SDR.
        """
        if input_value not in self._category_encodings:
            if self._new_categories_allowed and len(self._category_encodings) - 1 < self._max_categories:
                self._category_encodings[input_value] = self._allocate_slot()
                self._category_list.append(input_value)
                self.logger.info("Expanded category encoding for: %s", input_value)
            else:
                sdr = list(self._category_encodings[None])
                self.logger.info("Category encoded as unknown: %s", input_value)
                self._encoding_cache[input_value] = sdr
                return sdr

        sdr = list(self._category_encodings[input_value])
        self.logger.info("Category encoded value: %s", input_value)
        self._encoding_cache[input_value] = sdr
        return sdr

    @property
    def encoding_cache(self) -> dict[str, list[int]]:
        """Category-level cache mapping category strings to their SDR encodings."""
        return self._encoding_cache

    def register_encoding(self, category: str, encoded: list[int] | None = None) -> list[int]:
        """Cache an encoding for a category so decode can compare against it."""
        if encoded is not None:
            self._encoding_cache[category] = encoded
            return encoded
        return self.encode(category)

    def clear_registered_encodings(self) -> None:
        """Clear cached category encodings."""
        self._encoding_cache.clear()

    # TODO add candidates to this method
    @override
    def decode(
        self, input_sdr: list[int], candidates: Iterable[float] | None = None
    ) -> tuple[str | None, float]:
        """Decode an SDR back into its original category string.

        Converts the sparse representation back to the category string using
        the underlying encoder and category list mapping.

        Args:
            input_sdr: Binary SDR representation (list of 0s and 1s).
            candidates: Optional candidate values for decoding (not yet implemented).

        Returns:
            Tuple of (category_string, confidence_score) where confidence indicates
            the quality of the match.

        Note:
            Currently only implemented for RDSE-based encoders.
        """
        if not self._encoding_cache:
            return (None, 0.0)

        best_category: str | None = None
        best_overlap = -1

        for category, cached_sdr in self._encoding_cache.items():
            overlap = sum(1 for a, b in zip(input_sdr, cached_sdr) if a == 1 and b == 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_category = category

        active_bits = sum(1 for b in input_sdr if b == 1)
        confidence = best_overlap / active_bits if active_bits > 0 else 0.0

        self.logger.info("Decoded SDR into category: %s", best_category)
        return (best_category, confidence)

    def check_parameters(self, parameters: CategoryParametersNew) -> CategoryParametersNew:
        """Validate category encoder parameters.

        Args:
            parameters: The category parameters to validate.

        Returns:
            The validated parameters object.

        Raises:
            ValueError: If category_list is empty or contains duplicates.
        """
        if not parameters.category_list:
            raise ValueError("category_list cannot be empty.")
        if len(set(parameters.category_list)) != len(parameters.category_list):
            raise ValueError("category_list contains duplicate entries.")
        return parameters


@dataclass
class CategoryParametersNew:
    """Configuration parameters for CategoryEncoder.

    Attributes:
        active_bits_per_category: number of active bits per category.
        sparsity: the percent of sdr that is active bits.
        size: the size of the sdr.
        category_list: List of valid category strings to encode. Must be unique.
        new_categories_allowed: If True, unseen categories get unique encodings
            until capacity is exhausted, then fall back to unknown. If False,
            all unseen categories map to the unknown encoding.
        seed: Random seed for reproducible encodings.
        encoder_class: Reference to the CategoryEncoder class.
    """

    active_bits_per_category: int = 0
    sparsity: float = 0.02
    size: int = 2048
    category_list: list[Any] = field(default_factory=list)
    new_categories_allowed: bool = False
    seed: int = 42
    encoder_class = CategoryEncoderNew


if __name__ == "__main__":
    categories = ["ES", "GB", "US"]
    parameters = CategoryParametersNew(
        active_bits_per_category=0, category_list=categories
    )
    e = CategoryEncoderNew(parameters=parameters)
    a = e.encode("US")
    b = e.encode("ES")
    c = e.encode("NA")
    d = e.encode("GB")
    e.decode(a)

    e.decode(b)

    e.decode(c)

    e.decode(d)
    """
    categories = ["ES", "GB", "US"]
    parameters = CategoryParameters(w=3, category_list=categories)
    e1 = CategoryEncoder(parameters=parameters)
    a1 = SDR([1, 12])
    a2 = SDR([1, 12])
    e1.encode("ES", a1)
    e1.encode("ES", a2)
    print(a1.get_dense())
    print(a2.get_dense())
    assert a1.get_dense() == a2.get_dense()
    e1.encode("GB", a1)
    e1.encode("GB", a2)
    print(a1.get_dense())
    print(a2.get_dense())
    assert a1.get_dense() == a2.get_dense()
    e1.encode("US", a1)
    e1.encode("US", a2)
    print(a1.get_dense())
    print(a2.get_dense())
    assert a1.get_dense() == a2.get_dense()
    e1.encode("NA", a1)
    e1.encode("NA", a2)
    print(a1.get_dense())
    print(a2.get_dense())
    assert a1.get_dense() == a2.get_dense()
    """