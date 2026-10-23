import pytest

from traffic_violation.vlm.parsing import VlmParseError, parse_vlm_output


def test_parses_fenced_chatty_json() -> None:
    raw = """Analysis follows.
```json
{"violation_confirmed": true, "violation_type": "red_light",
 "confidence": 0.87, "explanation": "Vehicle crossed during red.",
 "visible_evidence": ["red lamp", "vehicle beyond line"]}
```
"""

    report = parse_vlm_output(raw)

    assert report.violation_confirmed is True
    assert report.violation_type == "red_light"
    assert report.confidence == 0.87


@pytest.mark.parametrize("confidence, expected", [(-0.2, 0.0), (1.7, 1.0)])
def test_clamps_numeric_confidence(confidence: float, expected: float) -> None:
    raw = (
        '{"violation_confirmed": false, "violation_type": null, '
        f'"confidence": {confidence}, "explanation": "Insufficient evidence"}}'
    )

    assert parse_vlm_output(raw).confidence == expected


@pytest.mark.parametrize("raw", ["not json", "[]", '{"confidence": 0.5}'])
def test_rejects_missing_or_non_object_reports(raw: str) -> None:
    with pytest.raises(VlmParseError, match="no valid report JSON"):
        parse_vlm_output(raw)
