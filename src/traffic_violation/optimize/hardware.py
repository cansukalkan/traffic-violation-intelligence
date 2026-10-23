"""Runtime hardware discovery and supported inference-path selection."""

from __future__ import annotations

import importlib.util

from traffic_violation.config import HardwareConfig


def detect_hardware() -> HardwareConfig:
    """Inspect CUDA without failing on CPU-only or minimal installations."""

    if importlib.util.find_spec("torch") is None:
        return HardwareConfig(gpu_name=None, vram_gb=0, paths=available_paths_cpu())

    import torch

    if not torch.cuda.is_available():
        return HardwareConfig(gpu_name=None, vram_gb=0, paths=available_paths_cpu())

    properties = torch.cuda.get_device_properties(0)
    has_tensorrt = importlib.util.find_spec("tensorrt") is not None
    hints = ["tensorrt"] if has_tensorrt else []
    hardware = HardwareConfig(
        gpu_name=str(properties.name),
        vram_gb=float(properties.total_memory) / (1024**3),
        paths=hints,
    )
    return hardware.model_copy(update={"paths": available_paths(hardware)})


def available_paths_cpu() -> list[str]:
    return ["pytorch_fp32", "onnxruntime_cpu"]


def available_paths(hardware: HardwareConfig) -> list[str]:
    """Return paths that are meaningful for the declared hardware."""

    if hardware.gpu_name is None or hardware.vram_gb <= 0:
        return available_paths_cpu()

    paths = ["pytorch_fp32", "pytorch_fp16", "onnxruntime_cuda"]
    if "tensorrt" in hardware.paths or any(
        path.startswith("tensorrt_") for path in hardware.paths
    ):
        paths.extend(["tensorrt_fp16", "tensorrt_int8"])
    return paths
