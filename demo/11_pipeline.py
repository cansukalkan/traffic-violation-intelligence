"""Offline end-to-end traffic candidate pipeline demonstration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from traffic_violation.config import load_site_config
from traffic_violation.detection.fake import FakeDetector
from traffic_violation.detection.types import Detection
from traffic_violation.pipeline.runner import run_pipeline
from traffic_violation.routing.policy import RoutingPolicy
from traffic_violation.rules.signal import SignalTracker
from traffic_violation.storage.sqlite import SQLiteEventStore
from traffic_violation.vlm.fake import FakeDescriber

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "demos" / "11_pipeline.txt"
VIDEO = ROOT / "reports" / "demos" / "11_pipeline.mp4"
PREVIEW = ROOT / "reports" / "demos" / "11_pipeline_preview.png"
DATABASE = ROOT / "reports" / "demos" / "11_pipeline.db"


def main() -> None:
    cfg = load_site_config(ROOT / "configs" / "site.yaml")
    bottoms = (350, 360, 370, 375, 385, 410)
    frames: list[tuple[float, np.ndarray]] = []
    script: list[list[Detection]] = []
    for index, bottom in enumerate(bottoms):
        frame = np.full((480, 640, 3), 38, dtype=np.uint8)
        frame[20:60, 500:540] = (0, 0, 255)
        frames.append((index * 0.2, frame))
        script.append([Detection("car", 0.95, (170, bottom - 60, 230, bottom))])

    responses = [
        '{"violation_confirmed": true, "violation_type": "red_light", '
        '"confidence": 0.94, "explanation": "Araç kırmızıda çizgiyi geçti."}',
        '{"violation_confirmed": true, "violation_type": "crosswalk", '
        '"confidence": 0.90, "explanation": "Araç yaya geçidine girdi."}',
    ]
    if DATABASE.exists():
        DATABASE.unlink()
    signal = SignalTracker(
        cfg.signal_roi,
        min_pixels=cfg.thresholds.signal_min_pixels,
        min_ratio=cfg.thresholds.signal_min_ratio,
    )
    with SQLiteEventStore(DATABASE) as store:
        stats = run_pipeline(
            frames,
            FakeDetector(script),
            lambda detections, _t, _index: [(7, item) for item in detections],
            signal,
            cfg,
            RoutingPolicy(0, 20, 0.0),
            FakeDescriber(responses),
            store,
        )
        events = store.query()

    writer = cv2.VideoWriter(str(VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (640, 480))
    if not writer.isOpened():
        raise RuntimeError("cannot create pipeline demo video")
    try:
        for index, (_, frame) in enumerate(frames):
            bottom = bottoms[index]
            cv2.line(frame, (0, 380), (640, 380), (255, 255, 255), 3)
            cv2.rectangle(frame, (170, bottom - 60), (230, bottom), (0, 200, 255), 3)
            cv2.putText(
                frame,
                f"frame={index} track=7",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            writer.write(frame)
    finally:
        writer.release()
    if not cv2.imwrite(str(PREVIEW), frames[-1][1]):
        raise RuntimeError("cannot create pipeline demo preview")

    confidences = ", ".join(
        str(event.vlm["confidence"]) for event in events if event.vlm
    )
    lines = [
        "DEMO 11 — Senkron uçtan uca boru hattı",
        f"Kare: {stats.frames}",
        f"İhlal adayı: {stats.violations}",
        f"VLM çağrısı: {stats.vlm_calls}",
        f"VLM atlandı: {stats.vlm_skipped}",
        f"SQLite kayıt: {len(events)}",
        f"Türler: {', '.join(event.kind for event in events)}",
        f"VLM güvenleri: {confidences}",
        f"Süre: {stats.elapsed_s:.4f} s",
        f"Video: {VIDEO}",
        f"Önizleme: {PREVIEW}",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Rapor: {REPORT}")


if __name__ == "__main__":
    main()
