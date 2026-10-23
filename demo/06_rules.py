"""Visual demonstration of deterministic violation candidate rules."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from traffic_violation.config import load_site_config
from traffic_violation.rules.signal import SignalTracker
from traffic_violation.rules.violations import evaluate
from traffic_violation.tracking.tracker import TrackState

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "demos" / "06_rules.png"


def main() -> None:
    cfg = load_site_config(ROOT / "configs" / "site.yaml")
    canvas = np.full((480, 640, 3), 42, dtype=np.uint8)
    polygon = np.array(cfg.zones["crosswalk"], dtype=np.int32)
    cv2.fillPoly(canvas, [polygon], (70, 70, 70))
    line = cfg.stop_lines["main"]
    cv2.line(canvas, line[0], line[1], (255, 255, 255), 4)

    movement = [(200, 350), (200, 420)]
    cv2.arrowedLine(canvas, movement[0], movement[1], (0, 180, 255), 5)
    track = TrackState(
        track_id=7,
        cls="car",
        history=[(0.4, 200.0, 350.0), (0.8, 200.0, 420.0)],
        frames_seen=8,
    )
    signal = SignalTracker((0, 0, 20, 20), min_pixels=1, min_ratio=0.0)
    signal.update(np.full((20, 20, 3), (0, 0, 255), dtype=np.uint8), 0.0)
    candidates = evaluate([track], signal, 0.8, cfg)

    cv2.circle(canvas, (600, 45), 20, (0, 0, 255), -1)
    cv2.putText(
        canvas,
        f"track=7 | {', '.join(item.kind for item in candidates)}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    encoded, payload = cv2.imencode(".png", canvas)
    if not encoded:
        raise RuntimeError(f"cannot write demo: {OUTPUT}")
    OUTPUT.write_bytes(payload.tobytes())
    print(f"Adaylar: {[item.kind for item in candidates]}")
    print(f"Görsel: {OUTPUT}")


if __name__ == "__main__":
    main()
