"""Public data types used by the RAG runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class Message(TypedDict):
    """Provider-neutral chat message."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class Document:
    """A document stored in the retrieval index."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A retrieved document and its similarity score."""

    document: Document
    score: float
    ranking_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class Citation:
    """Verified provenance for one exact context excerpt."""

    document_id: str
    source: str
    chunk_index: int | None
    score: float
    ranking_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    """Generated answer together with the exact retrieved sources."""

    text: str
    sources: tuple[SearchResult, ...]
    context_sources: tuple[SearchResult, ...] = ()
    citations: tuple[Citation, ...] = ()

    def __str__(self) -> str:
        return self.text
