"""Typed messages shared by asynchronous agent roles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from traffic_violation.detection.types import Detection
from traffic_violation.events import Event, ViolationCandidate
from traffic_violation.tracking.tracker import TrackState
from traffic_violation.vlm.parsing import VlmReport


@dataclass(frozen=True, slots=True)
class FrameMessage:
    sequence: int
    timestamp: float
    frame_bgr: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class DetectionsMessage:
    frame: FrameMessage
    detections: list[Detection]


@dataclass(frozen=True, slots=True)
class TracksMessage:
    frame: FrameMessage
    tracks: list[TrackState]


@dataclass(frozen=True, slots=True)
class ApprovedCandidate:
    candidate: ViolationCandidate
    approved_at: float


@dataclass(frozen=True, slots=True)
class AnalysisMessage:
    candidate: ViolationCandidate
    report: VlmReport


@dataclass(frozen=True, slots=True)
class EventMessage:
    event: Event


@dataclass(frozen=True, slots=True)
class MetricMessage:
    role: str
    latency_ms: float
    queue_depth: int
