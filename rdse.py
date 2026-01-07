import copy
import math
import random
import struct
from dataclasses import dataclass
from typing import List, override

import mmh3

from base_encoder import BaseEncoder

"""
 * Parameters for the RandomDistributedScalarEncoder (RDSE)
 *
 * Members "activeBits" & "sparsity" are mutually exclusive, specify exactly one
 * of them.
 *
 * Members "radius", "resolution", & "category" are mutually exclusive, specify
 * exactly one of them.
"""


@dataclass
class RDSEParameters:

    size: int = 0
    """
    * Member "size" is the total number of bits in the encoded output SDR.
    """
    active_bits: int = 0
    """
    * Member "activeBits" is the number of true bits in the encoded output SDR.
    """
    sparsity: float = 0
    """
    * Member "sparsity" is the fraction of bits in the encoded output which this
    * encoder will activate. This is an alternative way to specify the member
    * "activeBits".
    """
    radius: float = 0
    """
    * Member "radius" Two inputs separated by more than the radius have
    * non-overlapping representations. Two inputs separated by less than the
    * radius will in general overlap in at least some of their bits. You can
    * think of this as the radius of the input.
    """
    resolution: float = 0
    """
    * Member "resolution" Two inputs separated by greater than, or equal to the
    * resolution will in general have different representations.
    """
    category: bool = False
    """
    * Member "category" means that the inputs are enumerated categories.
    * If true then this encoder will only encode unsigned integers, and all
    * inputs will have unique / non-overlapping representations.
    """
    seed: int = 0
    """
    * Member "seed" forces different encoders to produce different outputs, even
    * if the inputs and all other parameters are the same.  Two encoders with the
    * same seed, parameters, and input will produce identical outputs.
    *
    * The seed 0 is special.  Seed 0 is replaced with a random number.
    """


"""
 * Encodes a real number as a set of randomly generated activations.
 *
 * Description:
 * The RandomDistributedScalarEncoder (RDSE) encodes a numeric scalar (floating
 * point) value into an SDR.  The RDSE is more flexible than the ScalarEncoder.
 * This encoder does not need to know the minimum and maximum of the input
 * range.  It does not assign an input->output mapping at construction.  Instead
 * the encoding is determined at runtime.
"""


class RandomDistributedScalarEncoder(BaseEncoder[float]):
    """Random Distributed Scalar Encoder (RDSE) implementation."""

    def __init__(self, parameters: RDSEParameters, dimensions: List[int] | None = None):
        self._parameters = copy.deepcopy(parameters)
        self._parameters = self.check_parameters(self._parameters)

        self._size = self._parameters.size
        self._active_bits = self._parameters.active_bits
        self._sparsity = self._parameters.sparsity
        self._radius = self._parameters.radius
        self._resolution = self._parameters.resolution
        self._category = self._parameters.category
        self._seed = self._parameters.seed

        super().__init__(dimensions, self._size)

    """
    Encodes an input value into an SDR with a random distributed scalar encoder.
    We employ the murmur hashing.
    """

    @override
    def encode(self, input_value: float) -> List[int]:
        if self._category:
            if input_value != int(input_value) or input_value < 0:
                raise ValueError("Input to category encoder must be an unsigned integer")

        data = [0] * self.size

        index = int(input_value / self._resolution)

        for offset in range(self._active_bits):
            hash_buffer = index + offset

            """
                The lower case i in the struct.pack makes this take in signed 32 bit integers.
                It is important to note the previous iteration used an upper case I which
                made this not take negative values. The struct.pack converts an integer
                into a byte representation.
            """
            bucket = mmh3.hash(struct.pack("i", hash_buffer), self._seed, signed=False)
            bucket = bucket % self.size
            """
                Don't worry about hash collisions.  Instead measure the critical
                properties of the encoder in unit tests and quantify how significant
                the hash collisions are.  This encoder can not fix the collisions
                because it does not record past encodings.  Collisions cause small
                deviations in the sparsity or semantic similarity, depending on how
                they're handled.
            """
            data[bucket] = 1
        return data

    # After encode we may need a check_parameters method since most of the encoders have this
    def check_parameters(self, parameters: RDSEParameters):
        assert parameters.size > 0

        num_active_args = 0
        if parameters.active_bits > 0:
            num_active_args += 1
        if parameters.sparsity > 0:
            num_active_args += 1

        assert num_active_args != 0, "Missing argument, need one of: 'activeBits' or 'sparsity'."
        assert (
            num_active_args == 1
        ), "Too many arguments, choose only one of: 'activeBits' or 'sparsity'."

        num_resolution_args = 0
        if parameters.radius > 0:
            num_resolution_args += 1
        if parameters.category:
            num_resolution_args += 1
        if parameters.resolution > 0:
            num_resolution_args += 1

        assert (
            num_resolution_args != 0
        ), "Missing argument, need one of: 'radius', 'resolution', 'category'."
        assert (
            num_resolution_args == 1
        ), "Too many arguments, choose only one of: 'radius', 'resolution', 'category'."

        args = parameters

        if args.sparsity > 0:
            assert 0 <= args.sparsity <= 1
            args.active_bits = int(round(args.size * args.sparsity))
            assert args.active_bits > 0

        if args.category:
            args.radius = 1

        if args.radius > 0:
            args.resolution = args.radius / args.active_bits
        elif args.resolution > 0:
            args.radius = args.active_bits * args.resolution

        while args.seed == 0:
            args.seed = random.getrandbits(32)

        return args

def sparsify(vector: List[int]) -> List[int]:
    """Converts a sparse activity vector to a activation list."""
    return [i for i, bit in enumerate(vector) if bit == 1]

if __name__ == "__main__":
    # Tests
    params = RDSEParameters(
        size=1000, active_bits=20, sparsity=0.0, radius=10, resolution=0, category=False, seed=1
    )
    e1 = RandomDistributedScalarEncoder(params)
    print(sparsify(e1.encode(10.0)))
    print(sparsify(e1.encode(20.0)))
    print(sparsify(e1.encode(1000.0)))
    """encoder = RandomDistributedScalarEncoder(params)
    output = SDR([encoder.size])
    encoder.encode(10.0, output)
    print(output.get_sparse())
    output2 = SDR([encoder.size])
    encoder.encode(10.0, output2)
    print(output2.get_sparse())"""