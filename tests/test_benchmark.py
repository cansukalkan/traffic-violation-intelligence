import numpy as np
import pytest

from traffic_violation.detection.fake import FakeDetector
from traffic_violation.optimize.benchmark import benchmark_detector


def test_percentiles_are_ordered() -> None:
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(30)]
    result = benchmark_detector(FakeDetector([[]] * 40), frames, warmup=5)
    assert result.fps > 0
    assert result.p95_ms >= result.p50_ms >= 0
    assert result.measured_frames == 30
    assert result.peak_mem_mb >= 0


def test_benchmark_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="frame"):
        benchmark_detector(FakeDetector([]), [], warmup=0)
    with pytest.raises(ValueError, match="warmup"):
        benchmark_detector(
            FakeDetector([]), [np.zeros((8, 8, 3), dtype=np.uint8)], warmup=-1
        )
