"""Hashable state representation built from binary vectors with distance support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class State:
    """Immutable, hashable collection of binary vectors."""

    vectors: tuple[tuple[bool], ...] = field(default_factory=tuple, repr=False)

    def __init__(self, vectors: dict[str, Sequence[bool]]) -> None:
        object.__setattr__(self, "vectors", tuple( tuple(vectors[state]) for state in sorted(vectors)))

    def __hash__(self) -> int:
        return hash(self.vectors)

    def _normalized_hamming_distance(self, other: State) -> float:
        """Calculate normalized Hamming distance between two states."""
        if len(self.vectors) != len(other.vectors):
            raise ValueError("States must have the same number of vectors")

        total_distance = 0.0
        total_length = 0

        for vec1, vec2 in zip(self.vectors, other.vectors):
            if len(vec1) != len(vec2):
                raise ValueError("Vectors must be of the same length")

            distance = sum(el1 != el2 for el1, el2 in zip(vec1, vec2))
            total_distance += distance
            total_length += len(vec1)

        return total_distance / total_length if total_length > 0 else 0.0

    def distance(self, other: State) -> float:
        return self._normalized_hamming_distance(other)
    
    def __str__(self) -> str:
        return str(hash(self))

    def __lt__(self, other):
        """Define a partial order based on subset relationships of vectors."""
        return all(set(vec1).issubset(set(vec2)) for vec1, vec2 in zip(self.vectors, other.vectors))
        