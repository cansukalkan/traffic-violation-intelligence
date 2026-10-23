"""Generate and classify a synthetic traffic-light sequence."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from traffic_violation.config import load_site_config
from traffic_violation.rules.signal import SignalState, SignalTracker

ROOT = Path(__file__).resolve().parents[1]
VIDEO_OUTPUT = ROOT / "reports" / "demos" / "03_signal.mp4"
PREVIEW_OUTPUT = ROOT / "reports" / "demos" / "03_signal_preview.png"
TEXT_OUTPUT = ROOT / "reports" / "demos" / "03_signal.txt"
ROI = (280, 40, 360, 260)


def _make_frame(active: SignalState) -> NDArray[np.uint8]:
    frame = np.full((360, 640, 3), 35, dtype=np.uint8)
    cv2.rectangle(frame, (280, 40), (360, 260), (15, 15, 15), thickness=-1)
    lamps = (
        (SignalState.RED, (320, 85), (0, 0, 255)),
        (SignalState.YELLOW, (320, 150), (0, 255, 255)),
        (SignalState.GREEN, (320, 215), (0, 255, 0)),
    )
    for state, center, color in lamps:
        shown = color if state is active else (30, 30, 30)
        cv2.circle(frame, center, 24, shown, thickness=-1)
    return frame


def main() -> None:
    site = load_site_config(ROOT / "configs" / "site.yaml")
    tracker = SignalTracker(
        ROI,
        min_pixels=site.thresholds.signal_min_pixels,
        min_ratio=site.thresholds.signal_min_ratio,
    )
    sequence = [SignalState.RED] * 6 + [SignalState.YELLOW] * 3 + [SignalState.GREEN] * 6
    VIDEO_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(VIDEO_OUTPUT), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (640, 360)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 video writer")

    rows = ["frame,t_s,expected,predicted"]
    preview_frames: list[NDArray[np.uint8]] = []
    try:
        for index, expected in enumerate(sequence):
            frame = _make_frame(expected)
            timestamp = index / 5.0
            predicted = tracker.update(frame, timestamp)
            color = (0, 210, 0) if predicted is expected else (0, 0, 255)
            cv2.rectangle(frame, ROI[:2], ROI[2:], (255, 255, 255), 2)
            cv2.putText(
                frame,
                f"expected={expected.value} predicted={predicted.value}",
                (40, 320),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            writer.write(frame)
            if index in {0, 6, 9}:
                preview_frames.append(cv2.resize(frame, (320, 180)))
            rows.append(f"{index},{timestamp:.1f},{expected.value},{predicted.value}")
    finally:
        writer.release()

    TEXT_OUTPUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    if not cv2.imwrite(str(PREVIEW_OUTPUT), cv2.hconcat(preview_frames)):
        raise RuntimeError("OpenCV could not write the signal preview")
    correct = sum(row.split(",")[-1] == row.split(",")[-2] for row in rows[1:])
    print(f"Sinyal demosu: {correct}/{len(sequence)} doğru sınıflandırma")
    print(f"Video: {VIDEO_OUTPUT}")
    print(f"Önizleme: {PREVIEW_OUTPUT}")
    print(f"Tablo: {TEXT_OUTPUT}")


if __name__ == "__main__":
    main()

