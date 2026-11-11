"""Small statistics helpers (no numpy dependency)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float | None:
    """Percentile with linear interpolation between closest ranks.

    Matches numpy's default ("linear") method. ``q`` is in [0, 100].
    Returns None for an empty sequence.
    """
    if not 0 <= q <= 100:
        raise ValueError(f"percentile q must be in [0, 100], got {q}")
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (q / 100) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    fraction = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def stdev(values: Sequence[float]) -> float | None:
    """Population standard deviation; None when fewer than two values."""
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / len(values))
