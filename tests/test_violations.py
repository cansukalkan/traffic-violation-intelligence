from pathlib import Path

import numpy as np

from traffic_violation.config import load_site_config
from traffic_violation.rules.signal import SignalTracker
from traffic_violation.rules.violations import evaluate
from traffic_violation.tracking.tracker import TrackState

ROOT = Path(__file__).resolve().parents[1]


def _signal(color_bgr: tuple[int, int, int], at: float = 0.0) -> SignalTracker:
    tracker = SignalTracker((0, 0, 20, 20), min_pixels=1, min_ratio=0.0)
    frame = np.full((20, 20, 3), color_bgr, dtype=np.uint8)
    tracker.update(frame, at)
    return tracker


def _crossing_track(frames_seen: int = 5) -> TrackState:
    return TrackState(
        track_id=7,
        cls="car",
        history=[(0.4, 200.0, 370.0), (0.6, 200.0, 410.0)],
        frames_seen=frames_seen,
    )


def test_red_light_crossing_emits_explainable_candidates() -> None:
    cfg = load_site_config(ROOT / "configs" / "site.yaml")
    candidates = evaluate([_crossing_track()], _signal((0, 0, 255)), 0.6, cfg)

    assert [candidate.kind for candidate in candidates] == ["red_light", "crosswalk"]
    assert candidates[0].track_id == 7
    assert candidates[0].evidence["line"] == "main"


def test_green_light_does_not_emit_red_light_candidate() -> None:
    cfg = load_site_config(ROOT / "configs" / "site.yaml")
    candidates = evaluate([_crossing_track()], _signal((0, 255, 0)), 0.6, cfg)

    assert all(candidate.kind != "red_light" for candidate in candidates)


def test_red_grace_and_short_tracks_are_ignored() -> None:
    cfg = load_site_config(ROOT / "configs" / "site.yaml")
    red = _signal((0, 0, 255), at=0.4)

    grace_candidates = evaluate([_crossing_track()], red, 0.6, cfg)
    assert all(candidate.kind != "red_light" for candidate in grace_candidates)
    assert not evaluate([_crossing_track(frames_seen=4)], _signal((0, 0, 255)), 0.6, cfg)
