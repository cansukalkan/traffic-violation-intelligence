"""HSV-based traffic light state estimation within a configured ROI."""

from __future__ import annotations

import math
from enum import StrEnum

import cv2
import numpy as np
from numpy.typing import NDArray

Roi = tuple[int, int, int, int]


class SignalState(StrEnum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


class SignalTracker:
    """Classify the dominant signal color in a rectangular image region."""

    def __init__(self, roi: Roi, min_pixels: int = 25, min_ratio: float = 0.02) -> None:
        x1, y1, x2, y2 = roi
        if x2 <= x1 or y2 <= y1:
            raise ValueError("roi must have positive width and height")
        if min_pixels < 1:
            raise ValueError("min_pixels must be positive")
        if not 0 <= min_ratio <= 1:
            raise ValueError("min_ratio must be between 0 and 1")
        self.roi = roi
        self.min_pixels = min_pixels
        self.min_ratio = min_ratio
        self.state = SignalState.UNKNOWN
        self.last_update_t: float | None = None
        self.state_since_t: float | None = None

    def update(self, frame_bgr: NDArray[np.uint8], t: float) -> SignalState:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame must be a three-channel BGR image")

        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self.roi
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if t < 0:
            raise ValueError("t cannot be negative")
        self.last_update_t = t
        if x1 >= x2 or y1 >= y2:
            self._set_state(SignalState.UNKNOWN, t)
            return self.state

        crop = frame_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        masks = {
            SignalState.RED: cv2.bitwise_or(
                cv2.inRange(hsv, (0, 80, 80), (10, 255, 255)),
                cv2.inRange(hsv, (170, 80, 80), (179, 255, 255)),
            ),
            SignalState.YELLOW: cv2.inRange(hsv, (15, 80, 80), (40, 255, 255)),
            SignalState.GREEN: cv2.inRange(hsv, (41, 60, 60), (95, 255, 255)),
        }
        scores = {state: int(cv2.countNonZero(mask)) for state, mask in masks.items()}
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        required = max(self.min_pixels, math.ceil(crop.shape[0] * crop.shape[1] * self.min_ratio))
        if ranked[0][1] < required or ranked[0][1] == ranked[1][1]:
            next_state = SignalState.UNKNOWN
        else:
            next_state = ranked[0][0]
        self._set_state(next_state, t)
        return self.state

    def _set_state(self, state: SignalState, t: float) -> None:
        if state is not self.state or self.state_since_t is None:
            self.state_since_t = t
        self.state = state

