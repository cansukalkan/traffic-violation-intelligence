import asyncio
from pathlib import Path

from traffic_violation.agents.orchestrator import run_role_pipeline
from traffic_violation.agents.pools import BoundedQueue, run_pool
from traffic_violation.agents.roles import RoleSpec, build_role_specs, route_candidates
from traffic_violation.config import load_site_config
from traffic_violation.events import ViolationCandidate
from traffic_violation.routing.policy import RoutingPolicy

ROOT = Path(__file__).resolve().parents[1]


def _candidate(track_id: int) -> ViolationCandidate:
    return ViolationCandidate("red_light", track_id, 0.0, "car", {})


def test_router_forwards_only_policy_approved_candidates() -> None:
    async def scenario() -> list[ViolationCandidate | None]:
        incoming: asyncio.Queue[ViolationCandidate | None] = asyncio.Queue()
        outgoing: asyncio.Queue[ViolationCandidate | None] = asyncio.Queue()
        for track_id in (1, 1, 2):
            await incoming.put(_candidate(track_id))
        await incoming.put(None)
        await route_candidates(incoming, outgoing, RoutingPolicy(5, 100, 0.0), lambda: 0.0)
        result: list[ViolationCandidate | None] = []
        while not outgoing.empty():
            result.append(outgoing.get_nowait())
        return result

    routed = [item for item in asyncio.run(scenario()) if item is not None]
    assert [item.track_id for item in routed] == [1, 2]


def test_generic_pool_processes_all_items() -> None:
    async def double(value: int) -> int:
        return value * 2

    result = asyncio.run(run_pool(range(30), double, size=3))
    assert sorted(result) == [value * 2 for value in range(30)]


def test_bounded_queue_drops_and_counts() -> None:
    queue = BoundedQueue[int](maxsize=2)
    for value in range(5):
        queue.put_or_drop(value)
    assert queue.dropped == 3
    assert queue.depth == 2


def test_config_maps_to_12_roles_and_31_workers() -> None:
    specs = build_role_specs(load_site_config(ROOT / "configs" / "site.yaml").pools)
    assert len(specs) == 12
    assert sum(spec.workers for spec in specs) == 31
    assert next(spec.workers for spec in specs if spec.name == "Observability") == 3


def test_orchestrator_reports_backpressure_and_metrics() -> None:
    async def identity(_role: str, item: int) -> int:
        await asyncio.sleep(0)
        return item

    roles = [RoleSpec("Reader", 2), RoleSpec("Writer", 1)]
    result = asyncio.run(run_role_pipeline(range(10), roles, identity, queue_size=3))

    assert len(result.outputs) <= 3
    assert result.metrics[0].dropped == 7
    assert sum(metric.processed for metric in result.metrics) >= len(result.outputs)
