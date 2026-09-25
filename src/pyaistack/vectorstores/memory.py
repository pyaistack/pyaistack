"""Thread-safe in-memory cosine-similarity vector store."""

from __future__ import annotations

import heapq
import math
import threading
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..exceptions import VectorStoreError
from ..rag.types import Document, SearchResult
from ._metadata import metadata_matches
from ._vectors import normalize as _normalize
from ._vectors import validate_search


@dataclass(frozen=True, slots=True)
class _StoredVector:
    document: Document
    vector: tuple[float, ...]


class InMemoryVectorStore:
    """Simple vector store for local development and the first framework version.

    Vectors are normalized when inserted, so retrieval only needs a dot product.
    A lock protects concurrent reads/writes inside one Python process.
    """

    def __init__(self) -> None:
        self._items: list[_StoredVector] = []
        self._dimension: int | None = None
        self._lock = threading.RLock()
        self._model_name: str | None = None

    def set_embedding_model(self, model_name: str) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise VectorStoreError("embedding model name cannot be empty")
        with self._lock:
            if self._model_name is not None and self._model_name != model_name:
                raise VectorStoreError("embedding model changed; clear the store first")
            self._model_name = model_name

    def add(self, documents: Sequence[Document], vectors: Sequence[Sequence[float]]) -> None:
        if len(documents) != len(vectors):
            raise VectorStoreError("documents and vectors must have the same length")
        if not documents:
            return

        normalized = [_normalize(vector) for vector in vectors]
        dimension = len(normalized[0])

        if any(len(vector) != dimension for vector in normalized):
            raise VectorStoreError("all embedding vectors must have the same dimension")

        with self._lock:
            if self._dimension is not None and dimension != self._dimension:
                raise VectorStoreError(
                    f"embedding dimension changed from {self._dimension} to {dimension}; "
                    "clear the store before changing embedding models"
                )
            self._dimension = dimension
            self._items.extend(
                _StoredVector(document=deepcopy(document), vector=vector)
                for document, vector in zip(documents, normalized, strict=True)
            )

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        min_score: float | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        validate_search(top_k, min_score)

        normalized_query = _normalize(query_vector)

        with self._lock:
            if not self._items:
                return []
            if self._dimension != len(normalized_query):
                raise VectorStoreError(
                    f"query dimension {len(normalized_query)} does not match index "
                    f"dimension {self._dimension}"
                )
            items = tuple(self._items)

        scored: list[tuple[float, int, Document]] = []
        for index, item in enumerate(items):
            if not metadata_matches(item.document.metadata, metadata_filter):
                continue
            score = math.fsum(a * b for a, b in zip(normalized_query, item.vector, strict=True))
            if min_score is None or score >= min_score:
                scored.append((score, index, item.document))

        best = heapq.nlargest(top_k, scored, key=lambda item: (item[0], -item[1]))
        return [
            SearchResult(document=deepcopy(document), score=score) for score, _, document in best
        ]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._dimension = None
            self._model_name = None

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
