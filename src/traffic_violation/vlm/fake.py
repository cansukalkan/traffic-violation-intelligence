"""Deterministic VLM replacement for tests and offline demos."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from traffic_violation.events import ViolationCandidate


class FakeDescriber:
    def __init__(self, responses: Iterable[str]) -> None:
        self._responses = deque(responses)

    def describe(
        self,
        frame_bgr: NDArray[np.uint8],
        candidate: ViolationCandidate,
    ) -> str:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame must be a three-channel BGR image")
        if not self._responses:
            raise RuntimeError("no scripted VLM response remains")
        return self._responses.popleft()
