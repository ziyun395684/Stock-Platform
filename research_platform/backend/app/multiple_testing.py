"""Small dependency-free multiple-testing helpers for factor reports."""
from __future__ import annotations

import math
from typing import Iterable


def normal_two_sided_p(value: float, sample_std: float, count: int) -> float:
    if count < 2 or not math.isfinite(value) or not math.isfinite(sample_std) or sample_std <= 0:
        return 1.0
    z = abs(value) / (sample_std / math.sqrt(count))
    return math.erfc(z / math.sqrt(2.0))


def benjamini_hochberg(p_values: Iterable[float]) -> list[float]:
    """Return q-values in the original order using BH step-up correction."""
    values = [min(max(float(value), 0.0), 1.0) for value in p_values]
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    q_values = [1.0] * len(values)
    running = 1.0
    total = len(values)
    for rank, (index, p_value) in reversed(list(enumerate(indexed, start=1))):
        running = min(running, p_value * total / rank)
        q_values[index] = min(running, 1.0)
    return q_values
