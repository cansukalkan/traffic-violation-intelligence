"""Validated configuration models loaded from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

Point = tuple[int, int]


class StrictModel(BaseModel):
    """Shared strictness rules for project configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Thresholds(StrictModel):
    min_track_frames: int = Field(ge=1)
    red_grace_seconds: float = Field(ge=0)
    signal_min_pixels: int = Field(default=25, ge=1)
    signal_min_ratio: float = Field(default=0.02, ge=0, le=1)


class SiteConfig(StrictModel):
    camera_id: str = Field(min_length=1)
    zones: dict[str, list[Point]]
    stop_lines: dict[str, list[Point]]
    signal_roi: tuple[int, int, int, int]
    thresholds: Thresholds
    pools: dict[str, int]

    @field_validator("zones")
    @classmethod
    def validate_zones(cls, value: dict[str, list[Point]]) -> dict[str, list[Point]]:
        if any(len(points) < 3 for points in value.values()):
            raise ValueError("each zone must contain at least three points")
        return value

    @field_validator("stop_lines")
    @classmethod
    def validate_stop_lines(cls, value: dict[str, list[Point]]) -> dict[str, list[Point]]:
        if any(len(points) != 2 for points in value.values()):
            raise ValueError("each stop line must contain exactly two points")
        return value

    @field_validator("signal_roi")
    @classmethod
    def validate_roi(cls, value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = value
        if x2 <= x1 or y2 <= y1:
            raise ValueError("signal_roi must have positive width and height")
        return value

    @field_validator("pools")
    @classmethod
    def validate_pools(cls, value: dict[str, int]) -> dict[str, int]:
        if any(size < 1 for size in value.values()):
            raise ValueError("pool sizes must be positive")
        return value


class ModelsConfig(StrictModel):
    yolo_weights: str = Field(min_length=1)
    vlm_id: str = Field(min_length=1)
    vlm_4bit: bool
    vlm_max_tokens: int = Field(gt=0)


class HardwareConfig(StrictModel):
    gpu_name: str | None = None
    vram_gb: float = Field(default=0, ge=0)
    paths: list[str] = Field(default_factory=list)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read configuration: {source}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in configuration: {source}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {source}")
    return raw


def load_site_config(path: str | Path) -> SiteConfig:
    return SiteConfig.model_validate(_load_yaml(path))


def load_models_config(path: str | Path) -> ModelsConfig:
    return ModelsConfig.model_validate(_load_yaml(path))


def load_hardware_config(path: str | Path) -> HardwareConfig:
    return HardwareConfig.model_validate(_load_yaml(path))

