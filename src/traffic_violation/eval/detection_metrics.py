"""Detection quality metrics isolated from the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DetectionMetrics:
    """Core object-detection metrics returned by Ultralytics validation."""

    map50: float
    map50_95: float
    precision: float
    recall: float


def evaluate_detection(
    weights: str,
    data_yaml: str | Path,
    device: str = "cpu",
    imgsz: int = 640,
) -> DetectionMetrics:
    """Run deterministic validation through a small project-owned interface."""

    if imgsz < 32:
        raise ValueError("imgsz must be at least 32")
    data_path = Path(data_yaml)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed; install traffic-violation[ml]") from exc
    model: Any = YOLO(weights)
    metrics = model.val(data=str(data_path), device=device, imgsz=imgsz, verbose=False)
    precision, recall, _, _ = metrics.box.mean_results()
    return DetectionMetrics(
        map50=float(metrics.box.map50),
        map50_95=float(metrics.box.map),
        precision=float(precision),
        recall=float(recall),
    )


def map50(
    weights: str,
    data_yaml: str | Path,
    device: str = "cpu",
    imgsz: int = 640,
) -> float:
    """Compatibility helper for callers that only need mAP@0.50."""

    return evaluate_detection(weights, data_yaml, device=device, imgsz=imgsz).map50

