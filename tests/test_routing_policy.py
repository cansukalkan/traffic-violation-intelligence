from traffic_violation.events import ViolationCandidate
from traffic_violation.routing.policy import RoutingPolicy


def _candidate(track_id: int = 1, confidence: float = 1.0) -> ViolationCandidate:
    return ViolationCandidate(
        "red_light", track_id, 0.0, "car", {"confidence": confidence}
    )


def test_first_candidate_is_routed() -> None:
    assert RoutingPolicy(5, 10, 0.0).should_call_vlm(_candidate(), 0.0) == (
        True,
        "approved",
    )


def test_same_track_and_kind_obeys_cooldown() -> None:
    policy = RoutingPolicy(5, 10, 0.0)
    policy.should_call_vlm(_candidate(), 0.0)

    assert policy.should_call_vlm(_candidate(), 2.0) == (False, "cooldown")
    assert policy.should_call_vlm(_candidate(), 5.0)[0] is True


def test_sliding_budget_expires_after_one_minute() -> None:
    policy = RoutingPolicy(0, 2, 0.0)
    assert policy.should_call_vlm(_candidate(1), 0.0)[0]
    assert policy.should_call_vlm(_candidate(2), 1.0)[0]
    assert policy.should_call_vlm(_candidate(3), 2.0) == (False, "budget")
    assert policy.should_call_vlm(_candidate(3), 60.0)[0]


def test_low_confidence_is_not_routed_or_charged_to_budget() -> None:
    policy = RoutingPolicy(0, 1, 0.75)

    assert policy.should_call_vlm(_candidate(1, 0.5), 0.0) == (False, "confidence")
    assert policy.should_call_vlm(_candidate(2, 0.9), 1.0)[0]
