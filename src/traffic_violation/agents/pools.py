"""Bounded queues and generic asynchronous worker pools."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Generic, TypeVar, cast

T = TypeVar("T")
R = TypeVar("R")
_STOP = object()


class BoundedQueue(Generic[T]):
    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be positive")
        self._queue: asyncio.Queue[T | object] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0
        self.max_depth = 0

    def put_or_drop(self, item: T) -> bool:
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self.dropped += 1
            return False
        self.max_depth = max(self.max_depth, self._queue.qsize())
        return True

    async def put(self, item: T) -> None:
        await self._queue.put(item)
        self.max_depth = max(self.max_depth, self._queue.qsize())

    async def get(self) -> T | object:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    async def close(self, worker_count: int) -> None:
        for _ in range(worker_count):
            await self._queue.put(_STOP)

    @property
    def depth(self) -> int:
        return self._queue.qsize()


def is_stop(item: object) -> bool:
    return item is _STOP


async def run_pool(
    items: Iterable[T],
    worker: Callable[[T], Awaitable[R]],
    size: int,
) -> list[R]:
    if size < 1:
        raise ValueError("size must be positive")
    queue: asyncio.Queue[T | object] = asyncio.Queue()
    results: list[R] = []
    for item in items:
        queue.put_nowait(item)
    for _ in range(size):
        queue.put_nowait(_STOP)

    async def consume() -> None:
        while True:
            item = await queue.get()
            try:
                if is_stop(item):
                    return
                results.append(await worker(cast(T, item)))
            finally:
                queue.task_done()

    await asyncio.gather(*(consume() for _ in range(size)))
    return results
