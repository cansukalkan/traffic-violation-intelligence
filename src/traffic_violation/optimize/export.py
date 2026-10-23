"""Explicit ONNX and TensorRT export boundaries."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


def export_onnx(
    weights: str | Path,
    out: str | Path,
    imgsz: int = 640,
    half: bool = False,
) -> str:
    if imgsz < 32:
        raise ValueError("imgsz must be at least 32")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required for ONNX export") from exc

    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model: Any = YOLO(str(weights))
    exported = Path(model.export(format="onnx", imgsz=imgsz, half=half, simplify=False))
    if exported.resolve() != destination.resolve():
        shutil.copy2(exported, destination)
    return str(destination)


def export_tensorrt(
    onnx_path: str | Path,
    out: str | Path,
    int8: bool = False,
    calib_dir: str | Path | None = None,
) -> str:
    source = Path(onnx_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    executable = shutil.which("trtexec")
    if executable is None:
        raise RuntimeError("TensorRT trtexec is not installed")
    if int8 and calib_dir is None:
        raise ValueError("INT8 export requires calibration data")

    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [executable, f"--onnx={source}", f"--saveEngine={destination}"]
    command.append("--int8" if int8 else "--fp16")
    subprocess.run(command, check=True, capture_output=True, text=True)
    if not destination.is_file():
        raise RuntimeError("TensorRT export did not create an engine")
    return str(destination)
