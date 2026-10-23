"""Dependency-free latency summaries for worker pools."""

from __future__ import annotations

from dataclasses import dataclass


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True, slots=True)
class PoolMetrics:
    role: str
    workers: int
    processed: int
    failed: int
    dropped: int
    max_queue_depth: int
    p50_ms: float
    p95_ms: float
