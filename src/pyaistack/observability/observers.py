"""Small standard-library observers for local inspection and application logging."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from .types import RAGEvent


class InMemoryObserver:
    """Thread-safe event collection for local inspection and tests."""

    def __init__(self) -> None:
        self._events: list[RAGEvent] = []
        self._lock = threading.RLock()

    @property
    def events(self) -> tuple[RAGEvent, ...]:
        """Return an immutable snapshot of collected events."""
        with self._lock:
            return tuple(self._events)

    def on_event(self, event: RAGEvent) -> None:
        with self._lock:
            self._events.append(event)

    def clear(self) -> None:
        """Remove collected events."""
        with self._lock:
            self._events.clear()


class LoggingObserver:
    """Write one sanitized JSON event through the standard logging system."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger if logger is not None else logging.getLogger("pyaistack.observability")

    def on_event(self, event: RAGEvent) -> None:
        self.logger.info("pyaistack_event=%s", json.dumps(event.to_dict(), sort_keys=True))


class JsonlObserver:
    """Append sanitized events to an application-selected JSON Lines file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._closed = False
        try:
            self._file = self.path.open("a", encoding="utf-8")
        except OSError as error:
            raise ValueError(f"could not open observability file: {self.path}") from error

    def on_event(self, event: RAGEvent) -> None:
        with self._lock:
            if self._closed:
                raise ValueError("JSONL observer is closed")
            self._file.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
            self._file.flush()

    def close(self) -> None:
        """Flush and close the event file. Calling this repeatedly is safe."""
        with self._lock:
            if not self._closed:
                self._file.close()
                self._closed = True

    def __enter__(self) -> JsonlObserver:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
