"""Deterministic candidate rules built on track histories and site geometry."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from traffic_violation.config import SiteConfig
from traffic_violation.events import ViolationCandidate
from traffic_violation.geometry import point_in_polygon, segments_intersect
from traffic_violation.rules.signal import SignalState
from traffic_violation.tracking.tracker import TrackState

_VEHICLES = {"bicycle", "car", "motorcycle", "bus", "truck"}


class SignalStatus(Protocol):
    state: SignalState
    state_since_t: float | None


def _eligible(track: TrackState, cfg: SiteConfig) -> bool:
    return (
        track.cls in _VEHICLES
        and track.frames_seen >= cfg.thresholds.min_track_frames
        and len(track.history) >= 2
    )


def evaluate(
    tracks: Iterable[TrackState],
    signal: SignalStatus,
    t: float,
    cfg: SiteConfig,
) -> list[ViolationCandidate]:
    """Return explainable candidates; downstream validation makes the final decision."""

    if t < 0:
        raise ValueError("t cannot be negative")
    candidates: list[ViolationCandidate] = []
    red_duration: float | None = None
    if signal.state is SignalState.RED and signal.state_since_t is not None:
        red_duration = t - signal.state_since_t
    red_ready = (
        red_duration is not None
        and red_duration >= cfg.thresholds.red_grace_seconds
    )

    for track in tracks:
        if not _eligible(track, cfg):
            continue
        previous = (track.history[-2][1], track.history[-2][2])
        current = (track.history[-1][1], track.history[-1][2])

        for line_name, line in cfg.stop_lines.items():
            if (
                red_ready
                and red_duration is not None
                and segments_intersect(previous, current, line[0], line[1])
            ):
                candidates.append(
                    ViolationCandidate(
                        kind="red_light",
                        track_id=track.track_id,
                        frame_t=t,
                        cls=track.cls,
                        evidence={
                            "rule": "red_signal_stop_line_crossing",
                            "line": line_name,
                            "signal": signal.state.value,
                            "red_duration_s": red_duration,
                            "movement": [previous, current],
                        },
                    )
                )

        for zone_name, polygon in cfg.zones.items():
            entered = not point_in_polygon(previous, polygon) and point_in_polygon(
                current, polygon
            )
            if entered:
                candidates.append(
                    ViolationCandidate(
                        kind="crosswalk",
                        track_id=track.track_id,
                        frame_t=t,
                        cls=track.cls,
                        evidence={
                            "rule": "vehicle_entered_crosswalk",
                            "zone": zone_name,
                            "movement": [previous, current],
                        },
                    )
                )
    return candidates
