"""Small deterministic PRNG with a single integer state.

SplitMix64 is used instead of :mod:`random` so the full generator state is one
integer that serializes trivially to JSON and can be reproduced bit-for-bit in
other languages (for example a browser client using BigInt).
"""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import TypeVar

_MASK = (1 << 64) - 1
T = TypeVar("T")


class Rng:
    """SplitMix64 generator."""

    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK

    def next_u64(self) -> int:
        """Return the next 64-bit unsigned integer."""
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def randbelow(self, n: int) -> int:
        """Return an integer in ``[0, n)``. Modulo bias is negligible for small ``n``."""
        if n <= 0:
            raise ValueError("n must be positive")
        return self.next_u64() % n

    def shuffle(self, seq: MutableSequence[T]) -> None:
        """Fisher-Yates shuffle in place."""
        for i in range(len(seq) - 1, 0, -1):
            j = self.randbelow(i + 1)
            seq[i], seq[j] = seq[j], seq[i]


def derive_seed(seed: int, stream: int) -> int:
    """Derive an independent seed for a named sub-stream."""
    return Rng((seed & _MASK) ^ ((stream * 0xD1B54A32D192ED03) & _MASK)).next_u64()
