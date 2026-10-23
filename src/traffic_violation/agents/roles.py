"""Logical role definitions and candidate router worker."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from traffic_violation.events import ViolationCandidate
from traffic_violation.routing.policy import RoutingPolicy


@dataclass(frozen=True, slots=True)
class RoleSpec:
    name: str
    workers: int


def build_role_specs(pools: dict[str, int]) -> list[RoleSpec]:
    """Map 14 configured pools to 12 logical roles and exactly 31 workers."""

    ordered = [
        "FrameReader",
        "Detector",
        "Tracker",
        "ViolationRule",
        "SignalWatcher",
        "Router",
        "Analyst",
        "Validator",
        "EventWriter",
        "Reporter",
    ]
    missing = [name for name in ordered if name not in pools]
    operations = ["Monitor", "Alerter", "Profiler"]
    missing.extend(name for name in operations if name not in pools)
    if "Evaluator" not in pools:
        missing.append("Evaluator")
    if missing:
        raise ValueError(f"missing worker pools: {', '.join(missing)}")
    specs = [RoleSpec(name, pools[name]) for name in ordered]
    specs.append(RoleSpec("Observability", sum(pools[name] for name in operations)))
    specs.append(RoleSpec("Evaluator", pools["Evaluator"]))
    return specs


async def route_candidates(
    input_queue: asyncio.Queue[ViolationCandidate | None],
    output_queue: asyncio.Queue[ViolationCandidate | None],
    policy: RoutingPolicy,
    now: Callable[[], float],
) -> None:
    while True:
        candidate = await input_queue.get()
        try:
            if candidate is None:
                await output_queue.put(None)
                return
            approved, _ = policy.should_call_vlm(candidate, now())
            if approved:
                await output_queue.put(candidate)
        finally:
            input_queue.task_done()
