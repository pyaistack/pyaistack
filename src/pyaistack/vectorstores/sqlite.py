"""Persistent exact cosine and optional FTS5 hybrid retrieval."""

from __future__ import annotations

import heapq
import json
import sqlite3
import struct
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..exceptions import VectorStoreError
from ..rag.types import Document, SearchResult
from ._hybrid import fuse, keyword_query
from ._metadata import metadata_matches
from ._sqlite_filters import compile_filter
from ._sqlite_fts import SCHEMA_VERSION, ensure_fts
from ._vectors import cosine, normalize, validate_search


class SQLiteVectorStore:
    """Local SQLite index. Exact vector scans remain linear in matching rows."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._closed = False
        self._model_name: str | None = None
        try:
            self._connection = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
            self._connection.create_function("pyaistack_matches", 2, self._matches)
            with self._connection:
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS pyaistack_metadata "
                    "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS pyaistack_vectors ("
                    "document_id TEXT PRIMARY KEY, text TEXT NOT NULL, "
                    "metadata_json TEXT NOT NULL, vector BLOB NOT NULL)"
                )
                version = self._get_metadata("fts_schema_version")
                if version is not None and version != SCHEMA_VERSION:
                    raise VectorStoreError("unsupported FTS schema version; upgrade PyAIStack")
        except (sqlite3.Error, VectorStoreError) as error:
            if hasattr(self, "_connection"):
                self._connection.close()
            raise VectorStoreError(f"could not open SQLite store: {error}") from error

    @staticmethod
    def _matches(metadata: str, encoded_filter: str) -> int:
        return int(metadata_matches(json.loads(metadata), json.loads(encoded_filter)))

    @contextmanager
    def _operation(self, *, write: bool = False) -> Iterator[None]:
        with self._lock:
            if self._closed:
                raise VectorStoreError("SQLite store is closed")
            try:
                # Also gives hybrid branches one consistent snapshot.
                self._connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                yield
                self._connection.commit()
            except (sqlite3.Error, ValueError, TypeError, OverflowError, struct.error) as error:
                self._connection.rollback()
                raise VectorStoreError(f"SQLite operation failed: {error}") from error
            except BaseException:
                self._connection.rollback()
                raise

    def _check_model(self, model: str) -> None:
        current = self._get_metadata("embedding_model")
        if current is not None and current != model:
            raise VectorStoreError(
                f"embedding model changed from {current} to {model}; clear the store first"
            )
        self._set_metadata("embedding_model", model)

    def set_embedding_model(self, model_name: str) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise VectorStoreError("embedding model name cannot be empty")
        with self._operation(write=True):
            self._check_model(model_name)
            self._model_name = model_name

    def add(self, documents: Sequence[Document], vectors: Sequence[Sequence[float]]) -> None:
        if len(documents) != len(vectors):
            raise VectorStoreError("documents and vectors must have the same length")
        if not documents:
            return
        normalized = [normalize(vector) for vector in vectors]
        dimension = len(normalized[0])
        if any(len(vector) != dimension for vector in normalized):
            raise VectorStoreError("all embedding vectors must have the same dimension")
        with self._operation(write=True):
            if self._model_name is not None:
                self._check_model(self._model_name)
            current = self._get_metadata("dimension")
            if current is not None and int(current) != dimension:
                raise VectorStoreError("embedding dimension changed; clear the store first")
            self._set_metadata("dimension", str(dimension))
            self._connection.executemany(
                "INSERT INTO pyaistack_vectors(document_id,text,metadata_json,vector) "
                "VALUES (?,?,?,?)",
                (
                    (
                        doc.id,
                        doc.text,
                        json.dumps(doc.metadata, allow_nan=False),
                        self._serialize(vector),
                    )
                    for doc, vector in zip(documents, normalized, strict=True)
                ),
            )

    def _check_query(self, query: Sequence[float]) -> bool:
        dimension = self._get_metadata("dimension")
        if dimension is None:
            return False
        if int(dimension) != len(query):
            raise VectorStoreError("query dimension does not match index dimension")
        if self._model_name is not None:
            current = self._get_metadata("embedding_model")
            if current != self._model_name:
                raise VectorStoreError("embedding model changed; reopen with the correct model")
        return True

    def _results(
        self,
        cursor: sqlite3.Cursor,
        query: Sequence[float],
    ) -> Iterator[SearchResult]:
        try:
            for row in cursor:
                metadata = json.loads(row[2])
                if not isinstance(metadata, dict):
                    raise VectorStoreError("stored metadata must be an object")
                yield SearchResult(
                    Document(row[0], row[1], metadata), cosine(query, self._deserialize(row[3]))
                )
        finally:
            cursor.close()

    def _vector_search(
        self,
        query: Sequence[float],
        top_k: int,
        sql_filter: tuple[str, list[Any]],
        min_score: float | None = None,
    ) -> list[SearchResult]:
        cursor = self._connection.execute(
            "SELECT document_id,text,metadata_json,vector FROM pyaistack_vectors "
            f"WHERE {sql_filter[0]}",
            sql_filter[1],
        )
        candidates = (
            result
            for result in self._results(cursor, query)
            if min_score is None or result.score >= min_score
        )
        # nsmallest streams rows and retains only top_k results.
        return heapq.nsmallest(top_k, candidates, key=lambda r: (-r.score, r.document.id))

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        min_score: float | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        validate_search(top_k, min_score)
        query = normalize(query_vector)
        encoded = compile_filter(metadata_filter)
        with self._operation():
            if not self._check_query(query):
                return []
            return self._vector_search(query, top_k, encoded, min_score)

    def hybrid_search(
        self,
        query: str,
        query_vector: Sequence[float],
        *,
        top_k: int,
        candidate_k: int = 20,
        min_score: float | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        validate_search(top_k, min_score)
        validate_search(candidate_k, None)
        if not isinstance(query, str) or not query.strip():
            raise VectorStoreError("query must be a non-empty string")
        vector = normalize(query_vector)
        encoded = compile_filter(metadata_filter)
        # Migration is opt-in and atomic, no re-embedding required.
        with self._operation(write=True):
            ensure_fts(self._connection)
        with self._operation():
            if not self._check_query(vector):
                return []
            limit = max(candidate_k, top_k)
            semantic = self._vector_search(vector, limit, encoded)
            lexical_query = keyword_query(query)
            if not lexical_query:
                return [r for r in semantic if min_score is None or r.score >= min_score][:top_k]
            cursor = self._connection.execute(
                "SELECT v.document_id,v.text,v.metadata_json,v.vector "
                "FROM pyaistack_fts JOIN pyaistack_vectors v ON v.rowid=pyaistack_fts.rowid "
                f"WHERE pyaistack_fts MATCH ? AND ({encoded[0]}) "
                "ORDER BY bm25(pyaistack_fts), v.document_id LIMIT ?",
                [lexical_query, *encoded[1], limit],
            )
            lexical = list(self._results(cursor, vector))
            return fuse(semantic, lexical, top_k, min_score)

    def clear(self) -> None:
        with self._operation(write=True):
            self._connection.execute("DELETE FROM pyaistack_vectors")
            self._connection.execute(
                "DELETE FROM pyaistack_metadata WHERE key IN ('dimension','embedding_model')"
            )
            self._model_name = None

    def __len__(self) -> int:
        with self._operation():
            return int(
                self._connection.execute("SELECT COUNT(*) FROM pyaistack_vectors").fetchone()[0]
            )

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> SQLiteVectorStore:
        if self._closed:
            raise VectorStoreError("SQLite store is closed")
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _get_metadata(self, key: str) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM pyaistack_metadata WHERE key=?", (key,)
        ).fetchone()
        return None if row is None else str(row[0])

    def _set_metadata(self, key: str, value: str) -> None:
        self._connection.execute(
            "INSERT INTO pyaistack_metadata(key,value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    @staticmethod
    def _serialize(vector: Sequence[float]) -> bytes:
        return struct.pack(f"!{len(vector)}d", *vector)

    @staticmethod
    def _deserialize(data: bytes) -> tuple[float, ...]:
        if not data or len(data) % 8:
            raise VectorStoreError("stored SQLite vector has invalid data")
        return struct.unpack(f"!{len(data) // 8}d", data)
