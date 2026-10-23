"""Demonstrate cooldown, budget and confidence routing decisions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from traffic_violation.events import ViolationCandidate
from traffic_violation.routing.policy import RoutingPolicy

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "demos" / "09_routing.txt"


def main() -> None:
    policy = RoutingPolicy(cooldown_s=5.0, budget_per_minute=12, min_confidence=0.6)
    reasons: Counter[str] = Counter()
    for index in range(100):
        candidate = ViolationCandidate(
            kind="red_light",
            track_id=index % 15,
            frame_t=index * 0.5,
            cls="car",
            evidence={"confidence": 0.4 if index % 7 == 0 else 0.9},
        )
        _, reason = policy.should_call_vlm(candidate, now=index * 0.5)
        reasons[reason] += 1

    lines = [
        "DEMO 09 — VLM yönlendirme politikası",
        "Toplam aday: 100",
        *[f"{reason}: {count}" for reason, count in sorted(reasons.items())],
        "Kural: cooldown -> dakikalık bütçe -> güven eşiği",
        "Sonuç: VLM kare başına çağrılmaz; yalnızca politika onaylı adayları görür.",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Rapor: {OUTPUT}")


if __name__ == "__main__":
    main()
