"""Persistence contract kept independent from SQLite and PostgreSQL."""

from __future__ import annotations

from typing import Protocol

from traffic_violation.events import Event, ViolationKind


class EventStore(Protocol):
    def append(self, event: Event) -> None: ...

    def query(
        self,
        *,
        kind: ViolationKind | None = None,
        camera_id: str | None = None,
        limit: int = 100,
    ) -> list[Event]: ...

    def close(self) -> None: ...
