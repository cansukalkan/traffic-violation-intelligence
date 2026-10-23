"""Domain events passed from deterministic rules to later analysis stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ViolationKind = Literal["red_light", "stop_line", "crosswalk"]


@dataclass(frozen=True, slots=True)
class ViolationCandidate:
    """A review candidate, not a final legal or enforcement decision."""

    kind: ViolationKind
    track_id: int
    frame_t: float
    cls: str
    evidence: dict[str, object]


@dataclass(frozen=True, slots=True)
class Event:
    """Persisted, auditable result of one deterministic rule candidate."""

    id: str
    camera_id: str
    kind: ViolationKind
    frame_t: float
    cls: str
    track_id: int
    evidence: dict[str, object]
    vlm: dict[str, object] | None
    created_at: float

    def __post_init__(self) -> None:
        if not self.id or not self.camera_id or not self.cls:
            raise ValueError("event id, camera_id and class cannot be empty")
        if self.frame_t < 0 or self.created_at < 0:
            raise ValueError("event timestamps cannot be negative")
        if self.track_id < 0:
            raise ValueError("track_id cannot be negative")
