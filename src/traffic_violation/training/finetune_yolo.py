"""Ultralytics fine-tuning adapter with early input validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def finetune(
    data_yaml: str | Path,
    base_weights: str,
    epochs: int,
    imgsz: int,
    device: str,
    *,
    project: str | Path = "runs/detect",
    name: str = "train",
    workers: int = 0,
) -> str:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if imgsz < 32:
        raise ValueError("imgsz must be at least 32")
    if workers < 0:
        raise ValueError("workers cannot be negative")
    data_path = Path(data_yaml)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed; install traffic-violation[ml]") from exc

    model: Any = YOLO(base_weights)
    result = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=str(project),
        name=name,
        workers=workers,
        seed=0,
        deterministic=True,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"training completed without expected best weights: {best}")
    return str(best)

