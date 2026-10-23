"""Track-history demo using deterministic ByteTrack-style IDs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from traffic_violation.detection.types import Detection
from traffic_violation.tracking.tracker import TrackRegistry, TrackState

ROOT = Path(__file__).resolve().parents[1]
VIDEO_OUTPUT = ROOT / "reports" / "demos" / "05_tracking.mp4"
PREVIEW_OUTPUT = ROOT / "reports" / "demos" / "05_tracking_preview.png"
TEXT_OUTPUT = ROOT / "reports" / "demos" / "05_tracking.txt"


def _draw_history(frame: np.ndarray, states: list[TrackState]) -> None:
    for state in states:
        points = np.array([(int(x), int(y)) for _, x, y in state.history], dtype=np.int32)
        if len(points) > 1:
            cv2.polylines(frame, [points], False, (255, 80, 30), 3)
        if len(points):
            cv2.putText(
                frame,
                f"id={state.track_id} {state.cls}",
                (int(points[-1][0]), max(25, int(points[-1][1]) - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )


def _run_synthetic() -> tuple[TrackRegistry, int, str]:
    width, height, fps, frame_count = 800, 480, 10.0, 40
    registry = TrackRegistry(max_missing=3, max_history=50)
    writer = cv2.VideoWriter(
        str(VIDEO_OUTPUT), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the tracking video writer")

    last_frame: np.ndarray | None = None
    try:
        for index in range(frame_count):
            frame = np.full((height, width, 3), 45, dtype=np.uint8)
            cv2.line(frame, (0, 350), (width, 350), (220, 220, 220), 3)
            x1 = 40 + index * 14
            detection = Detection("car", 0.94, (x1, 245, x1 + 120, 345))
            states = registry.update([(7, detection)], t=index / fps)
            state = states[0]
            cv2.rectangle(frame, (x1, 245), (x1 + 120, 345), (0, 210, 255), 3)
            _draw_history(frame, states)
            cv2.putText(
                frame,
                f"track_id={state.track_id} frames={state.frames_seen}",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            writer.write(frame)
            last_frame = frame
    finally:
        writer.release()
    if last_frame is None or not cv2.imwrite(str(PREVIEW_OUTPUT), last_frame):
        raise RuntimeError("OpenCV could not write the tracking preview")
    return registry, frame_count, "deterministic-offline"


def _run_real(source: Path, weights: str, device: str) -> tuple[TrackRegistry, int, str]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed; install traffic-violation[ml]") from exc

    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    model: Any = YOLO(weights)
    registry = TrackRegistry(max_missing=30, max_history=120)
    writer: cv2.VideoWriter | None = None
    last_frame: np.ndarray | None = None
    frame_count = 0
    try:
        results = model.track(
            source=str(source),
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",
            device=device,
            verbose=False,
        )
        for result in results:
            frame = result.orig_img.copy()
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(VIDEO_OUTPUT),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("OpenCV could not open the tracking video writer")

            observations: list[tuple[int, Detection]] = []
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.detach().cpu().tolist()
                coordinates = boxes.xyxy.detach().cpu().tolist()
                confidences = boxes.conf.detach().cpu().tolist()
                classes = boxes.cls.detach().cpu().tolist()
                for track_id, box, confidence, class_id in zip(
                    ids, coordinates, confidences, classes, strict=True
                ):
                    x1, y1, x2, y2 = (float(value) for value in box)
                    name = result.names[int(class_id)]
                    detection = Detection(str(name), float(confidence), (x1, y1, x2, y2))
                    observations.append((int(track_id), detection))
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 210, 255),
                        2,
                    )
            states = registry.update(observations, t=frame_count / fps)
            _draw_history(frame, states)
            writer.write(frame)
            last_frame = frame
            frame_count += 1
    finally:
        if writer is not None:
            writer.release()
    if frame_count == 0:
        raise RuntimeError(f"no frames could be read from {source}")
    if last_frame is None or not cv2.imwrite(str(PREVIEW_OUTPUT), last_frame):
        raise RuntimeError("OpenCV could not write the tracking preview")
    return registry, frame_count, "ultralytics-bytetrack"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    VIDEO_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if args.source:
        registry, frame_count, mode = _run_real(args.source, args.weights, args.device)
    else:
        registry, frame_count, mode = _run_synthetic()

    final = registry.active()[0]
    TEXT_OUTPUT.write_text(
        "\n".join(
            [
                "DEMO 05 — İz geçmişi",
                f"track_id: {final.track_id}",
                f"frames_seen: {final.frames_seen}",
                f"history_points: {len(final.history)}",
                f"processed_frames: {frame_count}",
                f"mode: {mode}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"İzleme demosu: track_id={final.track_id}, frames={final.frames_seen}")
    print(f"Video: {VIDEO_OUTPUT}")
    print(f"Önizleme: {PREVIEW_OUTPUT}")
    print(f"Rapor: {TEXT_OUTPUT}")


if __name__ == "__main__":
    main()

