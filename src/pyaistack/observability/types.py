"""Immutable, sanitized observability data objects."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any, Literal

EventStatus = Literal["success", "failure", "skipped"]
EVENT_SCHEMA_VERSION = 1


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class RAGEvent:
    """One completed RAG operation without prompts, answers, or document text."""

    run_id: str
    operation: str
    status: EventStatus
    duration_ms: float
    timestamp: str = field(default_factory=_utc_timestamp)
    schema_version: int = EVENT_SCHEMA_VERSION
    details: Mapping[str, Any] = field(default_factory=dict)
    error_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id cannot be empty")
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("operation cannot be empty")
        if self.status not in ("success", "failure", "skipped"):
            raise ValueError("status must be success, failure, or skipped")
        if not isinstance(self.duration_ms, (int, float)) or self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        if type(self.schema_version) is not int or self.schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {EVENT_SCHEMA_VERSION}")
        try:
            parsed_timestamp = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            raise ValueError("timestamp must be an ISO 8601 UTC timestamp") from None
        if parsed_timestamp.utcoffset() != timedelta(0):
            raise ValueError("timestamp must be an ISO 8601 UTC timestamp")
        try:
            encoded = json.dumps(dict(self.details), allow_nan=False, sort_keys=True)
            details = json.loads(encoded)
        except (TypeError, ValueError) as error:
            raise ValueError("event details must be JSON-compatible") from error
        object.__setattr__(self, "details", MappingProxyType(details))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible copy suitable for application-owned storage."""
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "operation": self.operation,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "details": dict(self.details),
            "error_type": self.error_type,
        }
