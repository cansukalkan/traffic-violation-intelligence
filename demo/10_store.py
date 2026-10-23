"""SQLite EventStore and timestamped frame-source demonstration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from traffic_violation.events import Event
from traffic_violation.pipeline.frame_source import frame_source
from traffic_violation.storage.sqlite import SQLiteEventStore

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "demos" / "10_store.txt"
DATABASE = ROOT / "reports" / "demos" / "10_events.db"
VIDEO = ROOT / "reports" / "demos" / "10_source.mp4"


def _make_video() -> None:
    writer = cv2.VideoWriter(str(VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (160, 96))
    if not writer.isOpened():
        raise RuntimeError("cannot create EventStore demo video")
    try:
        for index in range(5):
            frame = np.full((96, 160, 3), 30 + index * 25, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    _make_video()
    timestamps = [timestamp for timestamp, _ in frame_source(VIDEO)]
    if DATABASE.exists():
        DATABASE.unlink()
    with SQLiteEventStore(DATABASE) as store:
        kinds = ("red_light", "crosswalk", "red_light")
        for index, kind in enumerate(kinds):
            store.append(
                Event(
                    id=f"demo-{index}",
                    camera_id="cam-01",
                    kind=kind,
                    frame_t=index * 0.2,
                    cls="car",
                    track_id=index + 1,
                    evidence={"source": "synthetic-demo"},
                    vlm=None,
                    created_at=float(index),
                )
            )
        all_events = store.query()
        red_events = store.query(kind="red_light")

    lines = [
        "DEMO 10 — EventStore ve kare kaynağı",
        f"Video kareleri: {len(timestamps)}",
        f"Zamanlar: {', '.join(f'{value:.2f}' for value in timestamps)}",
        f"SQLite olayları: {len(all_events)}",
        f"red_light sorgusu: {len(red_events)}",
        f"Olay kimlikleri: {', '.join(event.id for event in all_events)}",
        f"Veritabanı: {DATABASE}",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Rapor: {REPORT}")


if __name__ == "__main__":
    main()
