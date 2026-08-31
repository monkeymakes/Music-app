"""Weighted reservoir sampling (Algorithm A-Res, Efraimidis & Spirakis).

Maintains a fixed-size sample from a stream where each item has a positive
weight. Higher weights are preferentially retained. Uses keys
``u ** (1 / weight)`` with ``u ~ Uniform(0, 1)`` and keeps the largest keys.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")

# Avoid log(0) / 0**inf edge cases when drawing keys.
_U_EPS = 1e-12


def reservoir_key(weight: float, u: float) -> float:
    """Compute the A-Res ranking key for ``weight`` and uniform draw ``u``."""
    if weight <= 0 or not math.isfinite(weight):
        raise ValueError("weight must be a finite number > 0")
    clamped = min(max(u, _U_EPS), 1.0 - _U_EPS)
    # Prefer exp(ln(u) / w) for numerical stability with large weights.
    return math.exp(math.log(clamped) / weight)


@dataclass(order=True)
class ReservoirEntry(Generic[T]):
    """Min-heap entry ordered by A-Res key (lowest key is eviction candidate)."""

    key: float
    seq: int
    weight: float = field(compare=False)
    item: T = field(compare=False)


class WeightedReservoir(Generic[T]):
    """Fixed-capacity weighted reservoir over an unbounded stream."""

    def __init__(self, capacity: int, *, rng: random.Random | None = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._rng = rng if rng is not None else random.Random()
        self._heap: list[ReservoirEntry[T]] = []
        self.seen = 0
        self.accepted = 0
        self.replaced = 0
        self._seq = 0

    def __len__(self) -> int:
        return len(self._heap)

    def offer(self, item: T, weight: float = 1.0) -> bool:
        """Offer ``item`` with ``weight``. Returns True if it is in the sample."""
        self.seen += 1
        self._seq += 1
        key = reservoir_key(weight, self._rng.random())
        entry = ReservoirEntry(key=key, seq=self._seq, weight=weight, item=item)

        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, entry)
            self.accepted += 1
            return True

        if key > self._heap[0].key:
            heapq.heapreplace(self._heap, entry)
            self.replaced += 1
            self.accepted += 1
            return True
        return False

    def items(self) -> list[T]:
        """Return sampled items in arrival order (stable for inference batches)."""
        return [entry.item for entry in sorted(self._heap, key=lambda e: e.seq)]

    def entries(self) -> list[ReservoirEntry[T]]:
        """Return full entries in arrival order."""
        return sorted(self._heap, key=lambda e: e.seq)

    def clear(self) -> None:
        """Drop sampled items and reset per-window counters."""
        self._heap.clear()
        self.seen = 0
        self.accepted = 0
        self.replaced = 0

    def stats(self) -> dict[str, int]:
        return {
            "capacity": self.capacity,
            "size": len(self._heap),
            "seen": self.seen,
            "accepted": self.accepted,
            "replaced": self.replaced,
        }
