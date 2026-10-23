"""Maintain domain-level histories for external tracker IDs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from traffic_violation.detection.types import Detection

HistoryPoint = tuple[float, float, float]


@dataclass(slots=True)
class TrackState:
    track_id: int
    cls: str
    history: list[HistoryPoint] = field(default_factory=list)
    frames_seen: int = 0
    missing_frames: int = 0


class TrackRegistry:
    def __init__(self, max_missing: int = 30, max_history: int = 120) -> None:
        if max_missing < 0:
            raise ValueError("max_missing cannot be negative")
        if max_history < 2:
            raise ValueError("max_history must be at least two")
        self.max_missing = max_missing
        self.max_history = max_history
        self._tracks: dict[int, TrackState] = {}

    def update(
        self,
        tracks: Iterable[tuple[int, Detection]],
        t: float,
    ) -> list[TrackState]:
        items = list(tracks)
        seen = [track_id for track_id, _ in items]
        if len(seen) != len(set(seen)):
            raise ValueError("duplicate track ID in the same frame")

        seen_ids = set(seen)
        for track_id, detection in items:
            if track_id < 0:
                raise ValueError("track ID cannot be negative")
            state = self._tracks.get(track_id)
            if state is None:
                state = TrackState(track_id=track_id, cls=detection.cls)
                self._tracks[track_id] = state
            center_x = (detection.box[0] + detection.box[2]) / 2.0
            bottom_y = detection.box[3]
            state.cls = detection.cls
            state.history.append((t, center_x, bottom_y))
            if len(state.history) > self.max_history:
                del state.history[: len(state.history) - self.max_history]
            state.frames_seen += 1
            state.missing_frames = 0

        expired: list[int] = []
        for track_id, state in self._tracks.items():
            if track_id not in seen_ids:
                state.missing_frames += 1
                if state.missing_frames > self.max_missing:
                    expired.append(track_id)
        for track_id in expired:
            del self._tracks[track_id]
        return self.active()

    def active(self) -> list[TrackState]:
        return sorted(
            (
                state
                for state in self._tracks.values()
                if state.missing_frames <= self.max_missing
            ),
            key=lambda state: state.track_id,
        )

