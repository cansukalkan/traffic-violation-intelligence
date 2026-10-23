"""Visual demo of polygon membership and segment intersection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from traffic_violation.geometry import point_in_polygon, segments_intersect

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "demos" / "02_geometry.png"


def main() -> None:
    canvas = np.full((520, 720, 3), 245, dtype=np.uint8)
    polygon = [(120.0, 100.0), (600.0, 100.0), (640.0, 410.0), (80.0, 410.0)]
    stop_line = ((60.0, 300.0), (660.0, 300.0))
    cv2.polylines(canvas, [np.array(polygon, dtype=np.int32)], True, (80, 80, 80), 3)
    cv2.line(canvas, tuple(map(int, stop_line[0])), tuple(map(int, stop_line[1])), (255, 80, 0), 3)

    rng = np.random.default_rng(42)
    inside_count = 0
    for x, y in rng.integers([25, 25], [695, 495], size=(20, 2)):
        inside = point_in_polygon((float(x), float(y)), polygon)
        inside_count += int(inside)
        color = (20, 170, 20) if inside else (30, 30, 220)
        cv2.circle(canvas, (int(x), int(y)), 7, color, thickness=-1)

    crossing = segments_intersect((360.0, 240.0), (360.0, 360.0), *stop_line)
    cv2.arrowedLine(canvas, (360, 240), (360, 360), (130, 0, 180), 3)
    cv2.putText(
        canvas,
        f"inside={inside_count}/20  path-crosses-line={crossing}",
        (40, 480),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (30, 30, 30),
        2,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    encoded, payload = cv2.imencode(".png", canvas)
    if not encoded:
        raise RuntimeError("OpenCV could not encode the demo image")
    OUTPUT.write_bytes(payload.tobytes())
    print(f"Geometri demosu: inside={inside_count}/20, crossing={crossing}")
    print(f"Görsel: {OUTPUT}")


if __name__ == "__main__":
    main()

