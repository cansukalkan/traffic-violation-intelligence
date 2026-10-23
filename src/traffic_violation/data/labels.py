"""Schema and loader for manually annotated violation events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ViolationKind = Literal["red_light", "stop_line", "crosswalk"]


class Label(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ViolationKind
    frame_t: float = Field(ge=0)
    track_id: int = Field(ge=0)
    note: str | None = None


def validate_label(data: dict[str, Any]) -> Label:
    """Validate one event record, normalizing Pydantic errors to ValueError."""

    try:
        return Label.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"invalid violation label: {exc}") from exc


def load_labels(path: str | Path) -> list[Label]:
    """Load and validate all events from a clip annotation JSON document."""

    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load label document: {source}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("events"), list):
        raise ValueError("label document must contain an events list")
    if not isinstance(document.get("clip"), str) or not document["clip"]:
        raise ValueError("label document must contain a non-empty clip name")
    fps = document.get("fps")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
        raise ValueError("label document fps must be positive")
    return [validate_label(event) for event in document["events"]]

