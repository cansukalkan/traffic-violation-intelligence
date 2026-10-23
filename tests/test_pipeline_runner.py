from pathlib import Path

import numpy as np

from traffic_violation.config import SiteConfig, Thresholds
from traffic_violation.detection.fake import FakeDetector
from traffic_violation.detection.types import Detection
from traffic_violation.events import ViolationCandidate
from traffic_violation.pipeline.runner import run_pipeline
from traffic_violation.routing.policy import RoutingPolicy
from traffic_violation.rules.signal import SignalState
from traffic_violation.storage.sqlite import SQLiteEventStore
from traffic_violation.vlm.fake import FakeDescriber


class AlwaysRed:
    state = SignalState.RED
    state_since_t: float | None = 0.0

    def update(self, frame_bgr: np.ndarray, t: float) -> SignalState:
        return self.state


def _config() -> SiteConfig:
    return SiteConfig(
        camera_id="cam-test",
        zones={},
        stop_lines={"main": [(0, 380), (640, 380)]},
        signal_roi=(0, 0, 10, 10),
        thresholds=Thresholds(min_track_frames=2, red_grace_seconds=0.0),
        pools={"Detector": 1},
    )


def test_end_to_end_pipeline_persists_parsed_vlm_report(tmp_path: Path) -> None:
    script = [
        [Detection("car", 0.9, (90, 330, 110, 370))],
        [Detection("car", 0.9, (90, 365, 110, 385))],
        [Detection("car", 0.9, (90, 380, 110, 400))],
    ]
    frames = [
        (index * 0.1, np.zeros((480, 640, 3), dtype=np.uint8))
        for index in range(3)
    ]
    response = (
        '{"violation_confirmed": true, "violation_type": "red_light", '
        '"confidence": 0.9, "explanation": "crossed on red"}'
    )
    with SQLiteEventStore(tmp_path / "events.db") as store:
        stats = run_pipeline(
            frames,
            FakeDetector(script),
            lambda detections, _t, _index: [(1, item) for item in detections],
            AlwaysRed(),
            _config(),
            RoutingPolicy(0, 100, 0.0),
            FakeDescriber([response]),
            store,
        )
        events = store.query()

    assert stats.frames == 3
    assert stats.violations == 1
    assert stats.vlm_calls == 1
    assert events[0].vlm is not None
    assert events[0].vlm["confidence"] == 0.9


def test_pipeline_records_skipped_candidate_without_vlm(tmp_path: Path) -> None:
    script = [
        [Detection("car", 0.9, (90, 330, 110, 370))],
        [Detection("car", 0.9, (90, 365, 110, 385))],
    ]
    frames = [(index * 0.1, np.zeros((480, 640, 3), dtype=np.uint8)) for index in range(2)]
    policy = RoutingPolicy(0, 1, 0.0)
    policy.should_call_vlm(ViolationCandidate("red_light", 99, 0.0, "car", {}), 0.0)
    with SQLiteEventStore(tmp_path / "events.db") as store:
        stats = run_pipeline(
            frames,
            FakeDetector(script),
            lambda detections, _t, _index: [(1, item) for item in detections],
            AlwaysRed(),
            _config(),
            policy,
            FakeDescriber([]),
            store,
        )
        events = store.query()

    assert stats.vlm_calls == 0
    assert stats.vlm_skipped == 1
    assert events[0].vlm is None
