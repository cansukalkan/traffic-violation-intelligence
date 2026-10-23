"""Concurrent 12-role reference orchestrator with bounded backpressure."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from traffic_violation.agents.monitor import PoolMetrics, percentile
from traffic_violation.agents.pools import BoundedQueue, is_stop
from traffic_violation.agents.roles import RoleSpec

T = TypeVar("T")
RoleHandler = Callable[[str, T], Awaitable[T]]


@dataclass(frozen=True, slots=True)
class OrchestratorResult(Generic[T]):
    outputs: list[T]
    metrics: list[PoolMetrics]


async def run_role_pipeline(
    items: Iterable[T],
    roles: list[RoleSpec],
    handler: RoleHandler[T],
    *,
    queue_size: int = 32,
) -> OrchestratorResult[T]:
    if not roles:
        raise ValueError("at least one role is required")
    queues = [BoundedQueue[T](queue_size) for _ in roles]
    outputs: list[T] = []
    latency: list[list[float]] = [[] for _ in roles]
    failures = [0 for _ in roles]
    processed = [0 for _ in roles]

    async def run_stage(index: int) -> None:
        spec = roles[index]
        input_queue = queues[index]
        output_queue = queues[index + 1] if index + 1 < len(queues) else None

        async def consume() -> None:
            while True:
                item = await input_queue.get()
                try:
                    if is_stop(item):
                        return
                    started = time.perf_counter()
                    try:
                        result = await handler(spec.name, cast(T, item))
                    except Exception:
                        failures[index] += 1
                        continue
                    latency[index].append((time.perf_counter() - started) * 1000.0)
                    processed[index] += 1
                    if output_queue is None:
                        outputs.append(result)
                    else:
                        output_queue.put_or_drop(result)
                finally:
                    input_queue.task_done()

        await asyncio.gather(*(consume() for _ in range(spec.workers)))
        if output_queue is not None:
            await output_queue.close(roles[index + 1].workers)

    stage_tasks = [asyncio.create_task(run_stage(index)) for index in range(len(roles))]
    for item in items:
        queues[0].put_or_drop(item)
    await queues[0].close(roles[0].workers)
    await asyncio.gather(*stage_tasks)

    metrics = [
        PoolMetrics(
            role=spec.name,
            workers=spec.workers,
            processed=processed[index],
            failed=failures[index],
            dropped=queues[index].dropped,
            max_queue_depth=queues[index].max_depth,
            p50_ms=percentile(latency[index], 0.50),
            p95_ms=percentile(latency[index], 0.95),
        )
        for index, spec in enumerate(roles)
    ]
    return OrchestratorResult(outputs=outputs, metrics=metrics)
