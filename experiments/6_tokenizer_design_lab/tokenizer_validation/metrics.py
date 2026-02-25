from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LengthStats:
    count: int
    mean: float
    p50: int
    p90: int
    p95: int
    p99: int
    max_len: int


def _percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return int(round(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac))


def length_stats(lengths: Iterable[int]) -> LengthStats:
    vals = [int(x) for x in lengths]
    vals.sort()
    if not vals:
        return LengthStats(count=0, mean=0.0, p50=0, p90=0, p95=0, p99=0, max_len=0)
    mean = sum(vals) / len(vals)
    return LengthStats(
        count=len(vals),
        mean=mean,
        p50=_percentile(vals, 0.50),
        p90=_percentile(vals, 0.90),
        p95=_percentile(vals, 0.95),
        p99=_percentile(vals, 0.99),
        max_len=vals[-1],
    )


def analyze_byte_fallback(input_ids: list[int], byte_ids: set[int]) -> float:
    """
    Returns the fraction of tokens that are byte fallbacks.
    """
    if not input_ids:
        return 0.0
        
    byte_count = 0
    for token_id in input_ids:
        if token_id in byte_ids:
            byte_count += 1
            
    return byte_count / len(input_ids)
