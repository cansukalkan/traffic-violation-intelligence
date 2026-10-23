"""Timestamped OpenCV frame iterator."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


def frame_source(path: str | Path) -> Iterator[tuple[float, NDArray[np.uint8]]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot open video source: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            position_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            fallback = frame_index / fps if fps > 0 else float(frame_index)
            timestamp = position_ms / 1000.0 if position_ms > 0 else fallback
            yield timestamp, cast(NDArray[np.uint8], frame)
            frame_index += 1
    finally:
        capture.release()
