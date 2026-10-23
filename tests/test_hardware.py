from traffic_violation.config import HardwareConfig
from traffic_violation.optimize.hardware import available_paths


def test_paths_without_gpu() -> None:
    hardware = HardwareConfig(gpu_name=None, vram_gb=0, paths=[])
    assert available_paths(hardware) == ["pytorch_fp32", "onnxruntime_cpu"]


def test_paths_with_gpu_without_tensorrt() -> None:
    hardware = HardwareConfig(gpu_name="RTX", vram_gb=8, paths=[])
    assert available_paths(hardware) == [
        "pytorch_fp32",
        "pytorch_fp16",
        "onnxruntime_cuda",
    ]


def test_paths_with_gpu_and_tensorrt() -> None:
    hardware = HardwareConfig(gpu_name="RTX", vram_gb=8, paths=["tensorrt"])
    assert available_paths(hardware)[-2:] == ["tensorrt_fp16", "tensorrt_int8"]
