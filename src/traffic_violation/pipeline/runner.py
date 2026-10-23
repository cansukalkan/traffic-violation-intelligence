"""Synchronous reference pipeline connecting all completed domain components."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from traffic_violation.config import SiteConfig
from traffic_violation.detection.types import Detection, Detector
from traffic_violation.events import Event
from traffic_violation.routing.policy import RoutingPolicy
from traffic_violation.rules.signal import SignalState
from traffic_violation.rules.violations import SignalStatus, evaluate
from traffic_violation.storage.protocol import EventStore
from traffic_violation.tracking.tracker import TrackRegistry
from traffic_violation.vlm.parsing import VlmParseError, parse_vlm_output
from traffic_violation.vlm.protocol import Describer

TrackerFunction = Callable[
    [list[Detection], float, int], Iterable[tuple[int, Detection]]
]


class SignalSource(SignalStatus, Protocol):
    def update(self, frame_bgr: NDArray[np.uint8], t: float) -> SignalState: ...


@dataclass(frozen=True, slots=True)
class PipelineStats:
    frames: int
    violations: int
    vlm_calls: int
    vlm_skipped: int
    elapsed_s: float


def run_pipeline(
    frames: Iterable[tuple[float, NDArray[np.uint8]]],
    detector: Detector,
    tracker_fn: TrackerFunction,
    signal: SignalSource,
    cfg: SiteConfig,
    policy: RoutingPolicy,
    describer: Describer,
    store: EventStore,
) -> PipelineStats:
    """Process timestamped frames and persist every explainable rule candidate."""

    started = time.perf_counter()
    registry = TrackRegistry()
    frame_count = 0
    violation_count = 0
    vlm_calls = 0
    vlm_skipped = 0

    for frame_index, (timestamp, frame) in enumerate(frames):
        if timestamp < 0:
            raise ValueError("frame timestamp cannot be negative")
        detections = detector.detect(frame)
        tracks = registry.update(tracker_fn(detections, timestamp, frame_index), timestamp)
        signal.update(frame, timestamp)
        candidates = evaluate(tracks, signal, timestamp, cfg)
        frame_count += 1

        for candidate in candidates:
            violation_count += 1
            approved, reason = policy.should_call_vlm(candidate, timestamp)
            vlm_payload: dict[str, object] | None = None
            evidence = dict(candidate.evidence)
            evidence["routing"] = reason
            if approved:
                vlm_calls += 1
                try:
                    report = parse_vlm_output(describer.describe(frame, candidate))
                    vlm_payload = report.model_dump(mode="json")
                except VlmParseError as exc:
                    evidence["vlm_parse_error"] = str(exc)
            else:
                vlm_skipped += 1
            store.append(
                Event(
                    id=uuid.uuid4().hex,
                    camera_id=cfg.camera_id,
                    kind=candidate.kind,
                    frame_t=candidate.frame_t,
                    cls=candidate.cls,
                    track_id=candidate.track_id,
                    evidence=evidence,
                    vlm=vlm_payload,
                    created_at=time.time(),
                )
            )

    return PipelineStats(
        frames=frame_count,
        violations=violation_count,
        vlm_calls=vlm_calls,
        vlm_skipped=vlm_skipped,
        elapsed_s=time.perf_counter() - started,
    )
