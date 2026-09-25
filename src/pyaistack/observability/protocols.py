"""Provider-neutral observability contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import RAGEvent


@runtime_checkable
class Observer(Protocol):
    """Receives one sanitized event after a RAG operation completes."""

    def on_event(self, event: RAGEvent) -> None: ...
