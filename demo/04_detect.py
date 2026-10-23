"""Detection demo with an offline fallback and optional real YOLO inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from traffic_violation.detection.fake import FakeDetector
from traffic_violation.detection.types import Detection, Detector
from traffic_violation.detection.yolo import YoloDetector

ROOT = Path(__file__).resolve().parents[1]
IMAGE_OUTPUT = ROOT / "reports" / "demos" / "04_detect.jpg"
TEXT_OUTPUT = ROOT / "reports" / "demos" / "04_finetune.txt"


def _read_image(path: Path) -> NDArray[np.uint8]:
    payload = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def _write_image(path: Path, image: NDArray[np.uint8]) -> None:
    encoded, payload = cv2.imencode(path.suffix, image)
    if not encoded:
        raise RuntimeError(f"cannot encode image as {path.suffix}")
    path.write_bytes(payload.tobytes())


def _synthetic() -> tuple[NDArray[np.uint8], Detector, str]:
    frame = np.full((480, 800, 3), (70, 70, 70), dtype=np.uint8)
    cv2.line(frame, (0, 330), (800, 330), (240, 240, 240), 4)
    cv2.rectangle(frame, (260, 230), (500, 370), (40, 120, 210), thickness=-1)
    scripted = [[Detection("car", 0.96, (260, 230, 500, 370))]]
    return frame, FakeDetector(scripted), "offline-fake"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.image:
        frame = _read_image(args.image)
        detector: Detector = YoloDetector(
            args.weights,
            {"car", "truck", "bus", "person", "traffic light"},
            device=args.device,
        )
        mode = "real-yolo"
    else:
        frame, detector, mode = _synthetic()

    detections = detector.detect(frame)
    for item in detections:
        x1, y1, x2, y2 = (int(value) for value in item.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 3)
        cv2.putText(
            frame,
            f"{item.cls} {item.conf:.2f}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 0),
            2,
        )
    IMAGE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _write_image(IMAGE_OUTPUT, frame)
    TEXT_OUTPUT.write_text(
        "\n".join(
            [
                "DEMO 04 — YOLO ince ayar durumu",
                f"Tespit modu: {mode}",
                f"Tespit sayısı: {len(detections)}",
                "İnce ayar: ÇALIŞTIRILMADI",
                "Neden: BDD100K lisans onayı ve veri yolu henüz sağlanmadı.",
                "Baseline/sonrası mAP50 veri hazır olduğunda aynı arayüzle ölçülecek.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Tespit demosu: mode={mode}, detections={len(detections)}")
    print(f"Görsel: {IMAGE_OUTPUT}")
    print(f"Eğitim durumu: {TEXT_OUTPUT}")


if __name__ == "__main__":
    main()

