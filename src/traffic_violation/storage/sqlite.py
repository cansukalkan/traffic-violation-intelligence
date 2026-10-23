"""Thread-safe SQLite implementation of the EventStore contract."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any, cast

from traffic_violation.events import Event, ViolationKind

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('red_light', 'stop_line', 'crosswalk')),
    frame_t REAL NOT NULL CHECK (frame_t >= 0),
    cls TEXT NOT NULL,
    track_id INTEGER NOT NULL CHECK (track_id >= 0),
    evidence_json TEXT NOT NULL,
    vlm_json TEXT,
    created_at REAL NOT NULL CHECK (created_at >= 0)
);
CREATE INDEX IF NOT EXISTS idx_events_kind_created ON events(kind, created_at);
CREATE INDEX IF NOT EXISTS idx_events_camera_created ON events(camera_id, created_at);
"""


class SQLiteEventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.executescript(_SCHEMA)

    def append(self, event: Event) -> None:
        try:
            evidence_json = json.dumps(event.evidence, ensure_ascii=False, sort_keys=True)
            vlm_json = (
                json.dumps(event.vlm, ensure_ascii=False, sort_keys=True)
                if event.vlm is not None
                else None
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("event evidence and VLM payload must be JSON serializable") from exc
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO events (
                    id, camera_id, kind, frame_t, cls, track_id,
                    evidence_json, vlm_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.camera_id,
                    event.kind,
                    event.frame_t,
                    event.cls,
                    event.track_id,
                    evidence_json,
                    vlm_json,
                    event.created_at,
                ),
            )

    def query(
        self,
        *,
        kind: ViolationKind | None = None,
        camera_id: str | None = None,
        limit: int = 100,
    ) -> list[Event]:
        if limit < 1:
            raise ValueError("limit must be positive")
        clauses: list[str] = []
        parameters: list[object] = []
        if kind is not None:
            clauses.append("kind = ?")
            parameters.append(kind)
        if camera_id is not None:
            clauses.append("camera_id = ?")
            parameters.append(camera_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT id, camera_id, kind, frame_t, cls, track_id, evidence_json, "
            f"vlm_json, created_at FROM events{where} ORDER BY created_at, id LIMIT ?"
        )
        parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(sql, parameters).fetchall()
        return [self._row_to_event(row) for row in rows]

    @staticmethod
    def _row_to_event(row: tuple[Any, ...]) -> Event:
        return Event(
            id=str(row[0]),
            camera_id=str(row[1]),
            kind=cast(ViolationKind, row[2]),
            frame_t=float(row[3]),
            cls=str(row[4]),
            track_id=int(row[5]),
            evidence=cast(dict[str, object], json.loads(row[6])),
            vlm=(
                cast(dict[str, object], json.loads(row[7]))
                if row[7] is not None
                else None
            ),
            created_at=float(row[8]),
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteEventStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
