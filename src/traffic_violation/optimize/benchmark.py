"""Repeatable detector latency and throughput measurements."""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from traffic_violation.detection.types import Detector


@dataclass(frozen=True, slots=True)
class BenchResult:
    fps: float
    p50_ms: float
    p95_ms: float
    peak_mem_mb: float
    measured_frames: int


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def benchmark_detector(
    detector: Detector,
    frames: Sequence[NDArray[np.uint8]],
    warmup: int = 5,
    *,
    synchronize: Callable[[], None] | None = None,
) -> BenchResult:
    """Benchmark one detector path after unmeasured warm-up calls.

    ``peak_mem_mb`` covers Python-managed allocations. Native and GPU memory must be
    collected by the serving runtime and reported separately.
    """

    if not frames:
        raise ValueError("at least one frame is required")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")

    for index in range(warmup):
        detector.detect(frames[index % len(frames)])
    if synchronize is not None:
        synchronize()

    timings_ms: list[float] = []
    tracemalloc.start()
    try:
        for frame in frames:
            started = time.perf_counter()
            detector.detect(frame)
            if synchronize is not None:
                synchronize()
            timings_ms.append((time.perf_counter() - started) * 1000.0)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    elapsed_s = sum(timings_ms) / 1000.0
    return BenchResult(
        fps=len(timings_ms) / elapsed_s if elapsed_s else float("inf"),
        p50_ms=_percentile(timings_ms, 0.50),
        p95_ms=_percentile(timings_ms, 0.95),
        peak_mem_mb=peak_bytes / (1024**2),
        measured_frames=len(timings_ms),
    )
