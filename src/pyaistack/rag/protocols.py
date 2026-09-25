"""RAG storage contract and compatibility exports for provider contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from ..providers.protocols import ChatProvider, EmbeddingProvider
from .types import Document, SearchResult

__all__ = ["ChatProvider", "EmbeddingProvider", "Reranker", "VectorStore"]


class VectorStore(Protocol):
    """Minimal vector-store contract used by RAG."""

    def add(self, documents: Sequence[Document], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        min_score: float | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]: ...

    def clear(self) -> None: ...

    def __len__(self) -> int: ...


@runtime_checkable
class ModelBoundStore(Protocol):
    """Optional index identity capability, rechecked before retrieval and writes."""

    def set_embedding_model(self, model_name: str) -> None: ...


@runtime_checkable
class HybridVectorStore(Protocol):
    """Optional keyword/vector retrieval capability."""

    def hybrid_search(
        self,
        query: str,
        query_vector: Sequence[float],
        *,
        top_k: int,
        candidate_k: int = 20,
        min_score: float | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]: ...


@runtime_checkable
class Reranker(Protocol):
    """Scores retrieved candidates in the same order they were supplied."""

    def rerank(self, query: str, candidates: Sequence[SearchResult]) -> Sequence[float]: ...
