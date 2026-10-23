import sqlite3
from pathlib import Path
from typing import cast

import pytest

from traffic_violation.events import Event, ViolationKind
from traffic_violation.pipeline.frame_source import frame_source
from traffic_violation.storage.sqlite import SQLiteEventStore


def _event(event_id: str, kind: str = "red_light") -> Event:
    return Event(
        event_id,
        "cam-01",
        cast(ViolationKind, kind),
        12.3,
        "car",
        4,
        {"line": "main"},
        {"confidence": 0.9},
        1.0,
    )


def test_sqlite_roundtrip(tmp_path: Path) -> None:
    with SQLiteEventStore(tmp_path / "events.db") as store:
        store.append(_event("e1"))
        event = store.query(kind="red_light")[0]

    assert event.track_id == 4
    assert event.vlm == {"confidence": 0.9}


def test_sqlite_filters_by_kind_and_camera(tmp_path: Path) -> None:
    with SQLiteEventStore(tmp_path / "events.db") as store:
        store.append(_event("a"))
        store.append(_event("b", "crosswalk"))

        assert [event.id for event in store.query(kind="crosswalk")] == ["b"]
        assert len(store.query(camera_id="cam-01")) == 2


def test_sqlite_rejects_duplicate_ids(tmp_path: Path) -> None:
    with SQLiteEventStore(tmp_path / "events.db") as store:
        store.append(_event("same"))
        with pytest.raises(sqlite3.IntegrityError):
            store.append(_event("same"))


def test_frame_source_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        next(frame_source(tmp_path / "missing.mp4"))
