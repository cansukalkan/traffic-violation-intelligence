"""Scripted detector used by tests and offline demos."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from traffic_violation.detection.types import Detection


class FakeDetector:
    def __init__(self, script: Sequence[Sequence[Detection]]) -> None:
        self._script = [list(frame) for frame in script]
        self._position = 0

    def detect(self, frame_bgr: NDArray[np.uint8]) -> list[Detection]:
        del frame_bgr
        if self._position >= len(self._script):
            return []
        detections = list(self._script[self._position])
        self._position += 1
        return detections

