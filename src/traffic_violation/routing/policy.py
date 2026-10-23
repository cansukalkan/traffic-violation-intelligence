"""Stateful VLM routing policy with cooldown and a sliding call budget."""

from __future__ import annotations

from collections import deque

from traffic_violation.events import ViolationCandidate


class RoutingPolicy:
    def __init__(
        self,
        cooldown_s: float,
        budget_per_minute: int,
        min_confidence: float,
    ) -> None:
        if cooldown_s < 0:
            raise ValueError("cooldown_s cannot be negative")
        if budget_per_minute < 1:
            raise ValueError("budget_per_minute must be positive")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self.cooldown_s = cooldown_s
        self.budget_per_minute = budget_per_minute
        self.min_confidence = min_confidence
        self._last_routed: dict[tuple[int, str], float] = {}
        self._calls: deque[float] = deque()

    def should_call_vlm(
        self,
        candidate: ViolationCandidate,
        now: float,
    ) -> tuple[bool, str]:
        if now < 0:
            raise ValueError("now cannot be negative")
        key = (candidate.track_id, candidate.kind)
        previous = self._last_routed.get(key)
        if previous is not None and now - previous < self.cooldown_s:
            return False, "cooldown"

        while self._calls and now - self._calls[0] >= 60.0:
            self._calls.popleft()
        if len(self._calls) >= self.budget_per_minute:
            return False, "budget"

        confidence_value = candidate.evidence.get("confidence", 1.0)
        confidence = (
            float(confidence_value)
            if isinstance(confidence_value, (int, float))
            and not isinstance(confidence_value, bool)
            else 0.0
        )
        if confidence < self.min_confidence:
            return False, "confidence"

        self._last_routed[key] = now
        self._calls.append(now)
        return True, "approved"
