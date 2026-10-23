"""Model-independent detection types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Detection:
    cls: str
    conf: float
    box: Box

    def __post_init__(self) -> None:
        if not self.cls:
            raise ValueError("class name cannot be empty")
        if not 0.0 <= self.conf <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        x1, y1, x2, y2 = self.box
        if x2 <= x1 or y2 <= y1:
            raise ValueError("detection box must have positive width and height")


class Detector(Protocol):
    """Contract shared by production and deterministic test detectors."""

    def detect(self, frame_bgr: NDArray[np.uint8]) -> list[Detection]: ...


def filter_classes(detections: list[Detection], allowed: set[str]) -> list[Detection]:
    """Keep detections whose normalized class name is explicitly allowed."""

    normalized = {name.casefold() for name in allowed}
    return [item for item in detections if item.cls.casefold() in normalized]

