from pathlib import Path

import pytest
from pydantic import ValidationError

from traffic_violation.config import load_hardware_config, load_models_config, load_site_config


def test_site_config(tmp_path: Path) -> None:
    path = tmp_path / "site.yaml"
    path.write_text(
        "camera_id: cam-01\n"
        "zones:\n  crosswalk: [[0, 400], [640, 400], [640, 460]]\n"
        "stop_lines:\n  main: [[0, 380], [640, 380]]\n"
        "signal_roi: [500, 20, 540, 60]\n"
        "thresholds:\n  min_track_frames: 5\n  red_grace_seconds: 0.5\n"
        "pools:\n  Detector: 4\n  Analyst: 4\n",
        encoding="utf-8",
    )
    config = load_site_config(path)
    assert config.stop_lines["main"][0] == (0, 380)
    assert config.pools["Detector"] == 4


def test_models_config_accepts_commercially_permissive_qwen3(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        "yolo_weights: yolo11n.pt\n"
        "vlm_id: Qwen/Qwen3-VL-4B-Instruct\n"
        "vlm_4bit: true\n"
        "vlm_max_tokens: 256\n",
        encoding="utf-8",
    )
    assert load_models_config(path).vlm_id.endswith("4B-Instruct")


def test_invalid_roi_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "site.yaml"
    path.write_text(
        "camera_id: c\nzones: {}\nstop_lines: {}\nsignal_roi: [5, 5, 1, 1]\n"
        "thresholds: {min_track_frames: 1, red_grace_seconds: 0}\npools: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_site_config(path)


def test_hardware_config_accepts_cpu_profile(tmp_path: Path) -> None:
    path = tmp_path / "hardware.yaml"
    path.write_text("gpu_name: null\nvram_gb: 0\npaths: [pytorch_fp32]\n", encoding="utf-8")
    assert load_hardware_config(path).gpu_name is None

