"""Benchmark hardware-selected detector paths with an offline reference fallback."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from traffic_violation.detection.fake import FakeDetector
from traffic_violation.detection.yolo import YoloDetector
from traffic_violation.optimize.benchmark import BenchResult, benchmark_detector
from traffic_violation.optimize.export import export_onnx
from traffic_violation.optimize.hardware import available_paths, detect_hardware

ROOT = Path(__file__).resolve().parents[1]
DEMO_REPORT = ROOT / "reports" / "demos" / "13_bench.txt"
CHART = ROOT / "reports" / "demos" / "13_bench.png"


@dataclass(frozen=True, slots=True)
class Row:
    path: str
    status: str
    result: BenchResult | None
    map_delta: str = "n/a"


def _read_frames(folder: Path, limit: int, imgsz: int) -> list[NDArray[np.uint8]]:
    frames: list[NDArray[np.uint8]] = []
    for path in sorted(folder.glob("*.jpg"))[:limit]:
        payload = np.fromfile(path, dtype=np.uint8)
        frame = cv2.imdecode(payload, cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(cv2.resize(frame, (imgsz, imgsz)))
    if not frames:
        raise ValueError(f"no readable JPG images in {folder}")
    return frames


def _reference() -> Row:
    frames = [np.zeros((320, 320, 3), dtype=np.uint8) for _ in range(80)]
    detector = FakeDetector([[]] * 100)
    return Row("protocol_reference", "measured", benchmark_detector(detector, frames, 10))


def _real_rows(
    paths: list[str],
    weights: str,
    image_dir: Path,
    limit: int,
    imgsz: int,
    repeats: int,
) -> list[Row]:
    frames = _read_frames(image_dir, limit, imgsz)
    measured_frames = frames * repeats
    rows: list[Row] = []
    for path in paths:
        try:
            if path == "pytorch_fp32":
                detector = YoloDetector(
                    weights, {"car", "truck", "bus", "person", "traffic light"}
                )
            elif path == "onnxruntime_cpu":
                onnx_path = ROOT / "reports" / "models" / f"yolo11n-{imgsz}.onnx"
                if not onnx_path.is_file():
                    export_onnx(weights, onnx_path, imgsz=imgsz, half=False)
                detector = YoloDetector(
                    str(onnx_path),
                    {"car", "truck", "bus", "person", "traffic light"},
                )
            else:
                rows.append(Row(path, "runtime unavailable", None))
                continue
            rows.append(Row(path, "measured", benchmark_detector(detector, measured_frames, 5)))
        except Exception as exc:
            rows.append(Row(path, f"failed: {type(exc).__name__}: {exc}", None))
    return rows


def _format_result(result: BenchResult | None) -> tuple[str, str, str, str]:
    if result is None:
        return "-", "-", "-", "-"
    return (
        f"{result.fps:.2f}",
        f"{result.p50_ms:.2f}",
        f"{result.p95_ms:.2f}",
        f"{result.peak_mem_mb:.2f}",
    )


def _write_chart(rows: list[Row]) -> None:
    measured = [row for row in rows if row.result is not None]
    image = np.full((140 + 80 * max(1, len(measured)), 1050, 3), 34, dtype=np.uint8)
    cv2.putText(
        image, "Detector throughput (same frame set)", (35, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2,
    )
    maximum = max((row.result.fps for row in measured if row.result), default=1.0)
    for index, row in enumerate(measured):
        assert row.result is not None
        y = 105 + index * 80
        width = int(500 * row.result.fps / maximum)
        cv2.putText(image, row.path, (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 2)
        cv2.rectangle(image, (250, y - 24), (250 + width, y + 8), (60, 180, 95), -1)
        cv2.putText(
            image, f"{row.result.fps:.2f} FPS | p95 {row.result.p95_ms:.2f} ms",
            (265 + width, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1,
        )
    if not cv2.imwrite(str(CHART), image):
        raise RuntimeError("could not write benchmark chart")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument(
        "--images", type=Path,
        default=ROOT / "data" / "coco128" / "images" / "train2017",
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.limit < 1 or args.repeats < 1:
        raise ValueError("limit and repeats must be positive")
    hardware = detect_hardware()
    paths = available_paths(hardware)
    rows = (
        _real_rows(
            paths,
            args.weights,
            args.images,
            args.limit,
            args.imgsz,
            args.repeats,
        )
        if args.real
        else [_reference()]
    )
    _write_chart(rows)

    lines = [
        "DEMO 13 — Donanım ve hız yolları",
        f"GPU: {hardware.gpu_name or 'yok'}",
        f"VRAM: {hardware.vram_gb:.2f} GB",
        f"Uygun yollar: {', '.join(paths)}",
        f"Mod: {'gerçek YOLO' if args.real else 'protokol referansı'}",
        f"Ölçülen kare/yol: {args.limit * args.repeats if args.real else 80}",
        "path | status | FPS | p50_ms | p95_ms | Python_peak_MB | mAP50_delta",
    ]
    for row in rows:
        fps, p50, p95, memory = _format_result(row.result)
        lines.append(
            f"{row.path} | {row.status} | {fps} | {p50} | {p95} | {memory} | {row.map_delta}"
        )
    lines.extend(
        [
            "Not: Python_peak_MB native/GPU belleğini kapsamaz.",
            "Not: mAP50 farkı BDD100K lisanslı değerlendirme kümesi sağlandığında ölçülecek.",
        ]
    )
    content = "\n".join(lines) + "\n"
    DEMO_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DEMO_REPORT.write_text(content, encoding="utf-8")
    dated = ROOT / "reports" / f"bench_{date.today().isoformat()}.md"
    dated.write_text("# Hardware benchmark\n\n```text\n" + content + "```\n", encoding="utf-8")
    print(content, end="")
    print(f"Grafik: {CHART}")
    print(f"Rapor: {dated}")


if __name__ == "__main__":
    main()
