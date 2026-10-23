"""Persistence interfaces and implementations."""

from traffic_violation.storage.protocol import EventStore
from traffic_violation.storage.sqlite import SQLiteEventStore

__all__ = ["EventStore", "SQLiteEventStore"]
