"""Provider contracts for model and embedding integrations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ..rag.types import Message


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Contract implemented by an embedding-model provider."""

    @property
    def model_name(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class ChatProvider(Protocol):
    """Contract implemented by a chat-model provider."""

    @property
    def model_name(self) -> str: ...

    def chat(self, messages: Sequence[Message]) -> str: ...
