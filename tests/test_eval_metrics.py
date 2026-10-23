import pytest

from traffic_violation.eval.plate_metrics import normalize_plate, plate_accuracy
from traffic_violation.eval.violation_metrics import match_events, precision_recall_f1


def test_match_within_tolerance() -> None:
    predictions = [
        {"kind": "red_light", "frame_t": 10.2},
        {"kind": "red_light", "frame_t": 50.0},
    ]
    gold = [{"kind": "red_light", "frame_t": 10.0}]
    summary = precision_recall_f1(match_events(predictions, gold, tolerance_s=0.5))
    assert summary["tp"] == 1
    assert summary["fp"] == 1
    assert summary["fn"] == 0
    assert summary["precision"] == 0.5
    assert summary["recall"] == 1.0


def test_match_is_one_to_one_and_kind_sensitive() -> None:
    predictions = [
        {"kind": "red_light", "frame_t": 1.0},
        {"kind": "red_light", "frame_t": 1.1},
        {"kind": "crosswalk", "frame_t": 1.0},
    ]
    gold = [{"kind": "red_light", "frame_t": 1.05}]
    result = match_events(predictions, gold, tolerance_s=0.1)
    assert result.tp == 1
    assert result.fp == 2
    assert result.fn == 0


def test_temporal_match_maximizes_cardinality() -> None:
    predictions = [
        {"kind": "red_light", "frame_t": 0.4},
        {"kind": "red_light", "frame_t": 0.0},
    ]
    gold = [
        {"kind": "red_light", "frame_t": 0.0},
        {"kind": "red_light", "frame_t": 0.8},
    ]
    assert match_events(predictions, gold, tolerance_s=0.5).tp == 2


def test_invalid_event_and_tolerance_are_rejected() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        match_events([], [], -0.1)
    with pytest.raises(ValueError, match="kind"):
        match_events([{"frame_t": 1.0}], [], 0.5)


def test_plate_normalization_and_accuracy() -> None:
    assert normalize_plate("34 abc 123") == "34ABC123"
    assert plate_accuracy(["34 abc 123", "06-X-1"], ["34ABC123", "06X2"]) == 0.5
    assert plate_accuracy([], []) == 1.0
