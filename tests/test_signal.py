import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from traffic_violation.rules.signal import SignalState, SignalTracker


def _frame(bgr: tuple[int, int, int], size: int = 50) -> NDArray[np.uint8]:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


@pytest.mark.parametrize(
    ("bgr", "expected"),
    [
        ((0, 0, 255), SignalState.RED),
        ((0, 255, 255), SignalState.YELLOW),
        ((0, 255, 0), SignalState.GREEN),
        ((5, 5, 5), SignalState.UNKNOWN),
    ],
)
def test_solid_signal_colors(bgr: tuple[int, int, int], expected: SignalState) -> None:
    assert SignalTracker((0, 0, 50, 50)).update(_frame(bgr), 0.0) is expected


def test_small_colored_noise_is_unknown() -> None:
    frame = _frame((0, 0, 0))
    frame[0:2, 0:2] = (0, 0, 255)
    tracker = SignalTracker((0, 0, 50, 50), min_pixels=10, min_ratio=0.01)
    assert tracker.update(frame, 0.0) is SignalState.UNKNOWN


def test_roi_is_clipped_to_frame() -> None:
    tracker = SignalTracker((-10, -10, 25, 25), min_pixels=5)
    assert tracker.update(_frame((0, 255, 0)), 0.0) is SignalState.GREEN


def test_empty_roi_is_unknown() -> None:
    tracker = SignalTracker((100, 100, 120, 120))
    assert tracker.update(_frame((0, 0, 255)), 0.0) is SignalState.UNKNOWN


def test_rejects_non_bgr_frame() -> None:
    tracker = SignalTracker((0, 0, 10, 10))
    with pytest.raises(ValueError, match="BGR"):
        tracker.update(np.zeros((10, 10), dtype=np.uint8), 0.0)


def test_realistic_lamp_on_dark_background() -> None:
    frame = _frame((0, 0, 0), size=80)
    cv2.circle(frame, (40, 20), 8, (0, 0, 255), thickness=-1)
    tracker = SignalTracker((20, 0, 60, 40), min_pixels=30, min_ratio=0.01)
    assert tracker.update(frame, 1.25) is SignalState.RED

