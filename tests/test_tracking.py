from traffic_violation.detection.types import Detection
from traffic_violation.tracking.tracker import TrackRegistry


def test_history_accumulates() -> None:
    registry = TrackRegistry()
    registry.update([(7, Detection("car", 0.9, (0, 0, 10, 20)))], t=0.0)
    states = registry.update([(7, Detection("car", 0.9, (0, 5, 10, 25)))], t=0.1)
    state = next(item for item in states if item.track_id == 7)
    assert state.frames_seen == 2
    assert state.history[-1][1:] == (5.0, 25)


def test_missing_becomes_inactive() -> None:
    registry = TrackRegistry(max_missing=2)
    registry.update([(1, Detection("car", 0.9, (0, 0, 1, 1)))], t=0.0)
    for timestamp in (0.1, 0.2, 0.3):
        registry.update([], t=timestamp)
    assert 1 not in [state.track_id for state in registry.active()]


def test_track_can_reappear_before_expiry() -> None:
    registry = TrackRegistry(max_missing=2)
    detection = Detection("car", 0.9, (0, 0, 10, 20))
    registry.update([(3, detection)], t=0.0)
    registry.update([], t=0.1)
    state = registry.update([(3, detection)], t=0.2)[0]
    assert state.frames_seen == 2
    assert state.missing_frames == 0


def test_class_change_updates_latest_class() -> None:
    registry = TrackRegistry()
    registry.update([(5, Detection("car", 0.8, (0, 0, 10, 20)))], t=0.0)
    state = registry.update([(5, Detection("truck", 0.9, (0, 0, 10, 20)))], t=0.1)[0]
    assert state.cls == "truck"


def test_duplicate_id_in_frame_is_rejected() -> None:
    registry = TrackRegistry()
    detection = Detection("car", 0.9, (0, 0, 10, 20))
    try:
        registry.update([(1, detection), (1, detection)], t=0.0)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate track IDs must be rejected")

