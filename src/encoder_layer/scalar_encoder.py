"""Scalar encoder for numeric values with semantic similarity.

This module provides a ScalarEncoder that converts numeric (floating point)
values into sparse distributed representations where semantically similar
values produce overlapping encodings. The encoder uses a contiguous block
of active bits whose position varies with the input value.

The encoder output is controlled by exactly one of four mutually exclusive
parameters: size, radius, category, or resolution.

Based on NuPIC's C++ implementation ported to Python.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Iterable, override

from encoder_layer.base_encoder import BaseEncoder, ParameterMarker
from log.log import get_logger, logger


class ScalarEncoder(BaseEncoder[int]):
    """Encoder for numeric values using contiguous blocks of active bits.

    The ScalarEncoder converts a numeric value into a sparse distributed
    representation where active bits form a contiguous block. The position
    of this block varies smoothly with the input value, ensuring semantic
    similarity - similar inputs produce overlapping representations.

    Args:
        parameters: Configuration for scalar encoding behavior.
    """

    def __init__(
        self,
        parameters: ScalarEncoderParameters,
    ):
        self._parameters = copy.deepcopy(parameters)
        self._parameters = self.check_parameters(self._parameters)

        self._minimum = self._parameters.minimum
        self._maximum = self._parameters.maximum
        self._clip_input = self._parameters.clip_input
        self._periodic = self._parameters.periodic
        self._category = self._parameters.category
        self._active_bits = self._parameters.active_bits
        self._sparsity = self._parameters.sparsity
        self._size = self._parameters.size
        self._radius = self._parameters.radius
        self._resolution = self._parameters.resolution
        self._encoding_cache: dict[float, list[int]] = {}
        self.logger = get_logger(self)
        super().__init__(self._size)

    """
        Encodes an input value into an SDR with a block of 1's.

        Description:
        The encode method is responsible for transforming the supplied SDR data structure into
        an SDR that has the encoding of the input value.
    """

    @property
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, value: int) -> None:

        self._size = value

        try:
            new_params = self.check_parameters(
                ScalarEncoderParameters(size=self._size, radius=0.0, category=False, resolution=0.0)
            )
            self._parameters = new_params

        except AssertionError as err:
            print(f"ERROR :: {self.__class__.__qualname__} :: {err}")
            self._size = self._parameters.size

    def register_encoding(self, input_value: float, encoded: list[int] | None = None) -> list[int]:
        """Caches an encoding so decode_closest can compare against it."""
        vector = encoded if encoded is not None else self._compute_encoding(input_value)
        if len(vector) != self.size:
            raise ValueError("Stored encoding must be the same length as encoder size")
        self._encoding_cache[input_value] = vector
        return vector

    def clear_registered_encodings(self) -> None:
        """Clears cached encodings used for nearest-neighbor decoding."""
        self._encoding_cache.clear()

    def _compute_encoding(self, input_value: int | float) -> list[int]:
        if self._clip_input:
            if self._periodic:

                input_value = input_value % self._maximum
            else:
                input_value = max(input_value, self._minimum)
                input_value = min(input_value, self._maximum)
        else:
            if self._category and input_value != float(int(input_value)):
                raise ValueError("Input to category encoder must be an unsigned integer!")
            if not (self._minimum <= input_value <= self._maximum):
                raise ValueError(
                    f"Input must be within range [{self._minimum}, {self._maximum}]! "
                    f"Received {input_value}"
                )

        start = int(round((input_value - self._minimum) / self._resolution))

        """Handle edge case where start + active_bits exceeds output size.
          // The endpoints of the input range are inclusive, which means that the
          // maximum value may round up to an index which is outside of the SDR. Correct
          // this by pushing the endpoint (and everything which rounds to it) onto the
          // last bit in the SDR.
        """

        if not self._periodic:
            start = min(start, self.size - self._active_bits)

        sparse: list[int] = []
        sparse[:] = range(start, start + self._active_bits)

        if self._periodic:
            for i, bit in enumerate(sparse):
                if bit >= self.size:
                    sparse[i] = bit - self.size.size
            sparse.sort()

        dense = [0] * self.size
        for idx in sparse:
            dense[idx] = 1

        return dense

    @override
    def encode(self, input_value: Any) -> list[int]:
        """Encode the input value into a binary vector."""
        if type(input_value) is not int and type(input_value) is not float:
            raise ValueError("A scalar encoder can only encode floats or ints.")
        self.register_encoding(input_value)
        self.logger.info("Scalar encoded value: %s", input_value)
        return self._compute_encoding(input_value)

    @override
    def decode(
        self, encoded: list[int], candidates: Iterable[float] | None = None
    ) -> tuple[float | None, float]:
        """Returns the value whose encoding overlaps the most with the provided SDR."""
        if len(encoded) != self.size:
            raise ValueError(
                f"Encoded input size ({len(encoded)}) does not match encoder size ({self.size})"
            )

        search_values = (
            list(candidates) if candidates is not None else list(self._encoding_cache.keys())
        )
        if not search_values:
            raise ValueError("No candidate encodings available for decoding")

        best_value: float | None = None
        best_overlap = -1

        for candidate in search_values:
            candidate_encoding = self._encoding_cache.get(candidate)
            if candidate_encoding is None:
                candidate_encoding = self.register_encoding(candidate)
            overlap = self._overlap(encoded, candidate_encoding)
            if overlap > best_overlap:
                best_overlap = overlap
                best_value = candidate

        confidence = (
            best_overlap / self._active_bits if best_overlap >= 0 and self._active_bits else 0.0
        )
        self.logger.info(
            "Scalar decoded SDR into value: %s, with confidence: %s", best_value, confidence
        )
        return best_value, confidence

    @staticmethod
    def _overlap(first: list[int], second: list[int]) -> int:
        """
        Checks for overlapping bits in two Lists of integers.

        :param first: The first list in checking.
        :type first: List[int]
        :param second: The second list in checking.
        :type second: List[int]
        :return: Returns the overlapping bits.
        :rtype: int
        """
        if len(first) != len(second):
            raise ValueError("Vectors must be the same length to compute overlap")
        return sum(1 for a, b in zip(first, second) if a == 1 and b == 1)

    def sparsify(self, vector: list[int]) -> list[int]:
        """Converts a sparse activity vector to a activation list."""
        return [i for i, bit in enumerate(vector) if bit == 1]

    # After encode we may need a check_parameters method since most of the encoders have this
    def check_parameters(self, parameters: "ScalarEncoderParameters"):
        """
        Check parameters method is used to verify that the correct parameters were entered
        and reject the user when they are not.

        Description: This changes and transforms the input that the user has with the parameters
        dataclass. There are many aspects such as the active bit and sparsity being mutually exclusive
        and the size, radius, resolution, and category also being muturally exclusive with each other.
        The user will have an assert that rejects when these are violated.
        """
        if not parameters.minimum <= parameters.maximum:
            raise ValueError("The minimum is not smaller than the maximum.")
        num_active_args = sum([parameters.active_bits > 0, parameters.sparsity > 0])
        if num_active_args == 0:
            raise ValueError("Missing argument, need one of: 'active_bits', 'sparsity'")
        if num_active_args != 1:
            raise ValueError("Specified both: 'active_bits', 'sparsity'. Specify only one of them.")
        num_size_args = sum(
            [
                parameters.radius > 0,
                parameters.category,
                parameters.resolution > 0,
            ]
        )
        if num_size_args == 0:
            raise ValueError("Missing argument, need one of: 'radius', 'resolution', 'category'.")
        if num_size_args != 1:
            raise ValueError(
                "Too many arguments specified: 'radius', 'resolution', 'category'. Choose only one of them."
            )
        if parameters.periodic:
            if parameters.clip_input:
                raise ValueError("Will not clip periodic inputs.  Caller must apply modulus.")
        if parameters.category:
            if parameters.clip_input:
                raise ValueError("Incompatible arguments: category & clip_input.")
            if parameters.periodic:
                raise ValueError("Incompatible arguments: category & periodic.")
            if not parameters.minimum == float(int(parameters.minimum)):
                raise ValueError(
                    "Minimum input value of category encoder must be an unsigned integer!"
                )
            if not parameters.maximum == float(int(parameters.maximum)):
                raise ValueError(
                    "Maximum input value of category encoder must be an unsigned integer!"
                )

        args = parameters
        if args.category:
            args.radius = 1.0
        if args.sparsity:
            if args.sparsity <= 0.0 or args.sparsity >= 1.0:
                raise ValueError("Sparsity should be between 0.0 and 1.0.")
            if not args.size > 0:
                raise ValueError("Argument 'sparsity' requires that the 'size' also be given.")
            args.active_bits = round(args.size * args.sparsity)
            if not args.active_bits > 0:
                raise ValueError("Sparsity and size must be given so that sparsity * size > 0!")
        if args.periodic:
            extent_width = args.maximum - args.minimum
        else:
            max_inclusive = math.nextafter(args.maximum, math.inf)
            extent_width = max_inclusive - args.minimum
        if args.size > 0:
            if args.periodic:
                args.resolution = extent_width / args.size
            else:
                n_buckets = args.size - (args.active_bits - 1)
                args.resolution = extent_width / (n_buckets - 1)
        else:
            if args.radius > 0.0:
                args.resolution = args.radius / args.active_bits

            needed_bands = math.ceil(extent_width / args.resolution)
            if args.periodic:
                args.size = needed_bands
            else:
                args.size = needed_bands + (args.active_bits - 1)

        # Sanity check the parameters.
        if not args.size > 0:
            raise ValueError("Sanity check for size failed, it was not greater than 0.")
        if not args.active_bits > 0:
            raise ValueError("Sanity check for active bits failed, it was not greater than 0.")
        if not args.active_bits < args.size:
            raise ValueError("Sanity check for active bits being less than size failed.")

        args.radius = args.active_bits * args.resolution
        if not args.radius > 0:
            raise ValueError("Sanity check failed for radius, it was not greater than 0.")

        args.sparsity = args.active_bits / float(args.size)
        if not args.sparsity > 0:
            raise ValueError("Sanity check failed for sparsity, it was not greater than 0.")

        return args


@dataclass
class ScalarEncoderParameters:
    """Configuration parameters for :class:`ScalarEncoder`."""

    minimum: int = 0
    """Min and Max
     * Members "minimum" and "maximum" define the range of the input signal.
     * These endpoints are inclusive.
     */"""
    maximum: int = 1000
    """Min and Max
     * Members "minimum" and "maximum" define the range of the input signal.
     * These endpoints are inclusive.
     */"""

    clip_input: bool = False
    """Whether to clip inputs outside the min/max range.
       /**
     * Member "clipInput" determines whether to allow input values outside the
     * range [minimum, maximum].
     * If true, the input will be clipped into the range [minimum, maximum].
     * If false, inputs outside of the range will raise an error.
     */"""

    periodic: bool = False
    """Whether the encoder is periodic (circular) or not.
    /**
     * Member "periodic" controls what happens near the edges of the input
     * range.
     *
     * If true, then the minimum & maximum input values are the same and the
     * first and last bits of the output SDR are adjacent.  The contiguous
     * block of 1's wraps around the end back to the beginning.
     *
     * If false, then minimum & maximum input values are the endpoints of the
     * input range, are not adjacent, and activity does not wrap around.
     */
    """

    category: bool = False
    """Whether the encoder is a category encoder.
    /**
     * Member "category" means that the inputs are enumerated categories.
     * If true then this encoder will only encode unsigned integers, and all
     * inputs will have unique / non-overlapping representations.
     */
    """
    active_bits: int = 40
    """Number of active bits in the output SDR.
     * Member "activeBits" is the number of true bits in the encoded output SDR.
     * The output encodings will have a contiguous block of this many 1's.
    """
    sparsity: float = 0.0
    """Sparsity level -- % of active bits.
    * Member "sparsity" is an alternative way to specify the member "activeBits".
    * Sparsity requires that the size to also be specified.
    * Specify only one of: activeBits or sparsity.
    """
    size: int = 2048
    """Total number of bits in the output SDR.
     /**
     * Member "size" is the total number of bits in the encoded output SDR.
     */
    """
    radius: float = 1.0
    """Approximate input range (width) covered by the active bits.
    /**
     * Member "radius" Two inputs separated by more than the radius have
     * non-overlapping representations. Two inputs separated by less than the
     * radius will in general overlap in at least some of their bits. You can
     * think of this as the radius of the input.
     */
    """
    resolution: float = 0.0
    """The smallest difference between two inputs that produces different outputs.
      /**
     * Member "resolution" Two inputs separated by greater than, or equal to the
     * resolution are guaranteed to have different representations.
     */"""
    encoder_class = ScalarEncoder


if __name__ == "__main__":
    # Example usage
    p = ScalarEncoderParameters(
        minimum=0,
        maximum=1000,
        clip_input=False,
        periodic=False,
        category=False,
        active_bits=1,
        sparsity=0.0,
        size=10,
        radius=0.0,
        resolution=1.0,
    )
    encoder = ScalarEncoder(p)
    e = encoder.encode(400)
    # print(e)
    d = encoder.decode(e)
    # print(d)