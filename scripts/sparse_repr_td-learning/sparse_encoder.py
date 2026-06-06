"""Sparse representation of discrete states — the stand-in for the HTM/SP output.

``UniqueSDREncoder`` gives every state a **distinct, non-overlapping** k-of-M binary
SDR (no shared active bits between states). This is the cleanest possible decoupling:
it carries *no* generalization between states, so the thesis TD value learner running
on top of it should reproduce tabular behaviour exactly — isolating the question
"does per-neuron value + weighted-mean + TD(lambda) actually work?" from any
representation-learning effects.

(One-hot is just the k=1 special case; this generalizes it to a distributed code.)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    """Make the project's real encoders (src/encoder_layer/) importable."""
    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


class UniqueSDREncoder:
    """state -> fixed k-of-M SDR, with disjoint active-bit sets across states."""

    def __init__(self, n_states: int, k: int = 8, seed: int = 0) -> None:
        self.n_states = n_states
        self.k = k
        self.n_bits = n_states * k  # exactly enough for fully-disjoint codes
        rng = random.Random(seed)
        bits = list(range(self.n_bits))
        rng.shuffle(bits)
        # Chunk the shuffled bit pool into n_states disjoint groups of k.
        self._sdr = [
            tuple(sorted(bits[i * k:(i + 1) * k])) for i in range(n_states)
        ]

    def encode(self, state: int) -> tuple[int, ...]:
        """Active-neuron indices for ``state`` (length k)."""
        return self._sdr[state]


class CategoryEncoderAdapter:
    """Wraps the project's real ``CategoryEncoderNew`` to the encode->indices API.

    CategoryEncoder treats states as *unrelated categories* and allocates each a
    disjoint bit block — i.e. non-overlapping SDRs, just like UniqueSDREncoder. So
    this should reproduce the tabular result. It's here to confirm the real encoder
    (not just my toy one) drives the thesis TD update correctly.
    """

    def __init__(self, n_states: int, k: int = 8, seed: int = 0) -> None:
        _add_src_to_path()
        from encoder_layer.category import CategoryEncoderNew, CategoryParametersNew

        # +1 unknown slot; ample headroom so allocation never falls back to "unknown".
        size = (n_states + 2) * k
        params = CategoryParametersNew(
            active_bits_per_category=k, sparsity=0.0, size=size,
            category_list=[str(s) for s in range(n_states)], seed=seed,
        )
        self._enc = CategoryEncoderNew(params)
        self.n_bits = self._enc.size

    def encode(self, state: int) -> tuple[int, ...]:
        dense = self._enc.encode(str(state))
        return tuple(i for i, b in enumerate(dense) if b == 1)


class RDSEEncoderAdapter:
    """Wraps the project's real ``RandomDistributedScalarEncoder`` (RDSE).

    RDSE encodes the state *index as a scalar*, so numerically-adjacent states share
    bits (a sliding hash window). On FrozenLake, index-adjacency is NOT value-adjacency
    (e.g. state 4 is on-path, state 5 is a hole), so this deliberately injects
    representational interference — the hard case for "does TD still hold?".

    ``resolution`` is the overlap knob: 1.0 → index-adjacent states share k-1 of k
    bits (heavy interference); ≤ 1/k → near non-overlapping (recovers tabular).
    """

    def __init__(self, n_states: int, k: int = 8, size: int = 256,
                 resolution: float = 1.0, seed: int = 1) -> None:
        _add_src_to_path()
        from encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters

        # RDSE treats seed 0 as "pick a random seed"; keep it nonzero for reproducibility.
        params = RDSEParameters(size=size, active_bits=k, sparsity=0.0,
                                resolution=resolution, seed=seed + 1)
        self._enc = RandomDistributedScalarEncoder(params)
        self.n_bits = size

    def encode(self, state: int) -> tuple[int, ...]:
        dense = self._enc.encode(float(state))
        return tuple(i for i, b in enumerate(dense) if b == 1)


def build_encoder(kind: str, n_states: int, k: int, seed: int,
                  rdse_size: int = 256, rdse_resolution: float = 1.0):
    """Factory: 'unique' | 'category' | 'rdse'."""
    if kind == "unique":
        return UniqueSDREncoder(n_states, k=k, seed=seed)
    if kind == "category":
        return CategoryEncoderAdapter(n_states, k=k, seed=seed)
    if kind == "rdse":
        return RDSEEncoderAdapter(n_states, k=k, size=rdse_size,
                                  resolution=rdse_resolution, seed=seed)
    raise ValueError(f"unknown encoder kind: {kind}")
