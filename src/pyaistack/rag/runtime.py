"""High-level Retrieval-Augmented Generation facade."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from time import perf_counter
from typing import Any
from uuid import uuid4

from ..chunking import TextChunker
from ..exceptions import (
    ConfigurationError,
    EmptyIndexError,
    ProviderError,
    RerankerError,
    VectorStoreError,
)
from ..loaders import LoadedDocument
from ..observability import Observer, RAGEvent
from ..providers.protocols import ChatProvider, EmbeddingProvider
from ..vectorstores import InMemoryVectorStore
from ..vectorstores._vectors import normalize
from .config import RAGConfig
from .context import append_citations, append_sources, build_citations, build_context
from .protocols import HybridVectorStore, ModelBoundStore, Reranker, VectorStore
from .types import Document, Message, RAGAnswer, SearchResult

LOGGER = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a grounded retrieval assistant.
Answer the user's question using only the supplied context.
If the context does not contain enough information, say that clearly.
Do not invent facts that are not supported by the context.
Do not create source citations or include source paths.
Applications can display verified details separately."""


class RAG:
    """Small production-oriented RAG runtime.

    Defaults to Ollama for embeddings and chat, while accepting injected provider
    implementations so storage/model backends can be swapped without changing
    application code.
    """

    def __init__(
        self,
        embedding_model: str = "embeddinggemma",
        llm_model: str = "gemma3:4b",
        ollama_host: str = "http://localhost:11434",
        *,
        embedding_provider: EmbeddingProvider | None = None,
        chat_provider: ChatProvider | None = None,
        vector_store: VectorStore | None = None,
        config: RAGConfig | None = None,
        chunker: TextChunker | None = None,
        reranker: Reranker | None = None,
        observers: Sequence[Observer] | None = None,
        system_prompt_suffix: str | None = None,
    ) -> None:
        if embedding_provider is None or chat_provider is None:
            from ..providers import OllamaChatProvider, OllamaEmbeddingProvider

        self.embedding_provider = (
            embedding_provider
            if embedding_provider is not None
            else OllamaEmbeddingProvider(
                model=embedding_model,
                host=ollama_host,
            )
        )
        self.chat_provider = (
            chat_provider
            if chat_provider is not None
            else OllamaChatProvider(
                model=llm_model,
                host=ollama_host,
            )
        )
        self.vector_store = vector_store if vector_store is not None else InMemoryVectorStore()
        self.config = config if config is not None else RAGConfig()
        if self.config.retrieval_mode == "hybrid" and not isinstance(
            self.vector_store, HybridVectorStore
        ):
            raise ConfigurationError(
                "hybrid retrieval requires a hybrid-capable store such as SQLiteVectorStore"
            )
        self.chunker = chunker
        self.reranker = reranker
        self._observers = self._validate_observers(observers)
        if system_prompt_suffix is not None and (
            not isinstance(system_prompt_suffix, str) or not system_prompt_suffix.strip()
        ):
            raise ValueError("system_prompt_suffix cannot be empty")
        self.system_prompt_suffix = (
            system_prompt_suffix.strip() if system_prompt_suffix is not None else None
        )
        self._bind_model()

    def _bind_model(self) -> None:
        if isinstance(self.vector_store, ModelBoundStore):
            self.vector_store.set_embedding_model(self.embedding_provider.model_name)

    @property
    def document_count(self) -> int:
        return len(self.vector_store)

    def add(
        self,
        texts: str | Sequence[str],
        *,
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[Document, ...]:
        """Embed and index one or more text documents."""

        run_id = self._new_run_id()
        operation_started = perf_counter()
        values = [texts] if isinstance(texts, str) else list(texts)
        if not values:
            self._emit(run_id, "indexing", "skipped", operation_started, {"document_count": 0})
            return ()

        if any(not isinstance(text, str) for text in values):
            raise ValueError("texts must contain strings")
        cleaned = [text.strip() for text in values]
        if any(not text for text in cleaned):
            raise ValueError("texts cannot contain empty values")

        if metadatas is None:
            metadata_values: list[Mapping[str, Any]] = [{} for _ in cleaned]
        else:
            metadata_values = list(metadatas)
            if len(metadata_values) != len(cleaned):
                raise ValueError("metadatas must have the same length as texts")

        documents = tuple(
            Document(id=str(uuid4()), text=text, metadata=deepcopy(dict(metadata)))
            for text, metadata in zip(cleaned, metadata_values, strict=True)
        )
        try:
            for document in documents:
                json.dumps(document.metadata, allow_nan=False)
                if any(not isinstance(key, str) for key in document.metadata):
                    raise ValueError("metadata keys must be strings")
        except (TypeError, ValueError) as error:
            raise ValueError(
                "metadatas must be JSON-compatible mappings with string keys"
            ) from error

        self._bind_model()
        embedding_started = perf_counter()
        try:
            embeddings = self.embedding_provider.embed(cleaned)
        except Exception as error:
            self._emit_failure(run_id, "embedding", embedding_started, error)
            self._emit_failure(run_id, "add", operation_started, error)
            raise
        try:
            for vector in embeddings:
                normalize(vector)
        except (VectorStoreError, TypeError) as error:
            self._emit_failure(run_id, "embedding", embedding_started, error)
            self._emit_failure(run_id, "add", operation_started, error)
            raise ProviderError("embedding provider returned malformed vectors") from error
        if len(embeddings) != len(documents):
            self._emit_failure(
                run_id,
                "embedding",
                embedding_started,
                ProviderError("embedding provider returned an unexpected vector count"),
            )
            self._emit_failure(
                run_id,
                "add",
                operation_started,
                ProviderError("embedding provider returned an unexpected vector count"),
            )
            raise ProviderError(
                f"embedding provider returned {len(embeddings)} vectors for "
                f"{len(documents)} documents"
            )

        self._emit(
            run_id,
            "embedding",
            "success",
            embedding_started,
            {"input_count": len(cleaned), "model": self.embedding_provider.model_name},
        )

        self._bind_model()
        indexing_started = perf_counter()
        try:
            self.vector_store.add(documents, embeddings)
        except Exception as error:
            self._emit_failure(run_id, "indexing", indexing_started, error)
            self._emit_failure(run_id, "add", operation_started, error)
            raise
        details = self._store_details(
            self._metadata_details({"document_count": len(documents)}, documents)
        )
        self._emit(run_id, "indexing", "success", indexing_started, details)
        self._emit(run_id, "add", "success", operation_started, {"document_count": len(documents)})
        LOGGER.info(
            "Indexed %s document(s) using embedding model %s",
            len(documents),
            self.embedding_provider.model_name,
        )
        return documents

    def add_documents(self, documents: Sequence[LoadedDocument]) -> tuple[Document, ...]:
        """Index documents loaded from files, applying the configured chunker when present."""

        values = tuple(documents)
        if self.chunker is not None:
            values = self.chunker.chunk(values)
        return self.add(
            [document.text for document in values],
            metadatas=[document.metadata for document in values],
        )

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> tuple[SearchResult, ...]:
        """Retrieve documents most similar to the query and metadata filter."""

        run_id = self._new_run_id()
        started = perf_counter()
        try:
            results = self._search(
                query,
                top_k=top_k,
                metadata_filter=metadata_filter,
                run_id=run_id,
            )
        except Exception as error:
            self._emit_failure(run_id, "search", started, error)
            raise
        self._emit(
            run_id,
            "search",
            "success",
            started,
            self._store_details({"result_count": len(results)}),
        )
        return results

    def _search(
        self,
        query: str,
        *,
        top_k: int | None,
        metadata_filter: Mapping[str, Any] | None,
        run_id: str | None,
    ) -> tuple[SearchResult, ...]:
        """Retrieve results for one public search or ask run."""

        if not isinstance(query, str):
            raise ValueError("query must be a string")
        question = query.strip()
        if not question:
            raise ValueError("query cannot be empty")
        if self.document_count == 0:
            raise EmptyIndexError("no documents are indexed; call rag.add(...) first")

        limit = self.config.top_k if top_k is None else top_k
        if type(limit) is not int or limit <= 0:
            raise ValueError("top_k must be greater than zero")
        if metadata_filter is not None and not isinstance(metadata_filter, Mapping):
            raise ValueError("metadata_filter must be a mapping")
        if metadata_filter is not None:
            try:
                json.dumps(dict(metadata_filter), allow_nan=False)
                if any(not isinstance(key, str) for key in metadata_filter):
                    raise ValueError("filter keys must be strings")
            except (TypeError, ValueError) as error:
                raise ValueError("metadata_filter must be JSON-compatible") from error

        self._bind_model()
        embedding_started = perf_counter()
        try:
            vectors = self.embedding_provider.embed([question])
        except Exception as error:
            self._emit_failure(run_id, "embedding", embedding_started, error)
            raise
        if not isinstance(vectors, Sequence) or len(vectors) != 1:
            self._emit_failure(
                run_id,
                "embedding",
                embedding_started,
                ProviderError("embedding provider returned an unexpected query vector count"),
            )
            raise ProviderError("embedding provider must return exactly one query vector")
        try:
            normalize(vectors[0])
        except VectorStoreError as error:
            self._emit_failure(run_id, "embedding", embedding_started, error)
            raise ProviderError("embedding provider returned an invalid query vector") from error
        self._emit(
            run_id,
            "embedding",
            "success",
            embedding_started,
            {"input_count": 1, "model": self.embedding_provider.model_name},
        )

        candidate_limit = (
            max(limit, self.config.rerank_candidate_k) if self.reranker is not None else limit
        )

        retrieval_started = perf_counter()
        try:
            if self.config.retrieval_mode == "hybrid":
                if not isinstance(self.vector_store, HybridVectorStore):
                    raise ConfigurationError("store does not support hybrid retrieval")
                results = tuple(
                    self.vector_store.hybrid_search(
                        question,
                        vectors[0],
                        top_k=candidate_limit,
                        candidate_k=max(candidate_limit, self.config.candidate_k),
                        min_score=self.config.min_score,
                        metadata_filter=metadata_filter,
                    )
                )
            else:
                results = tuple(
                    self.vector_store.search(
                        vectors[0],
                        top_k=candidate_limit,
                        min_score=self.config.min_score,
                        metadata_filter=metadata_filter,
                    )
                )
        except Exception as error:
            self._emit_failure(run_id, "retrieval", retrieval_started, error)
            raise
        self._emit(
            run_id,
            "retrieval",
            "success",
            retrieval_started,
            self._store_details(
                {
                    "mode": self.config.retrieval_mode,
                    "result_count": len(results),
                    "metadata_filter_keys": sorted(metadata_filter) if metadata_filter else [],
                }
            ),
        )
        return self._rerank(question, results, limit, run_id=run_id)

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> RAGAnswer:
        """Retrieve context and ask the configured LLM for a grounded answer."""

        run_id = self._new_run_id()
        answer_started = perf_counter()
        if not isinstance(question, str):
            raise ValueError("question must be a string")
        query = question.strip()
        if not query:
            raise ValueError("question cannot be empty")

        try:
            sources = self._search(
                query,
                top_k=top_k,
                metadata_filter=metadata_filter,
                run_id=run_id,
            )
        except Exception as error:
            self._emit_failure(run_id, "answer", answer_started, error)
            raise
        if len(sources) < self.config.min_context_results:
            return self._insufficient_context_answer(sources, run_id, answer_started)

        context_started = perf_counter()
        context, context_sources = build_context(sources, self.config.max_context_chars)
        self._emit(
            run_id,
            "context",
            "success",
            context_started,
            {"context_char_count": len(context), "context_source_count": len(context_sources)},
        )
        if len(context_sources) < self.config.min_context_results:
            return self._insufficient_context_answer(sources, run_id, answer_started)
        messages: list[Message] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"},
        ]
        generation_started = perf_counter()
        try:
            answer = self.chat_provider.chat(messages)
        except Exception as error:
            self._emit_failure(run_id, "generation", generation_started, error)
            self._emit_failure(run_id, "answer", answer_started, error)
            raise
        if not isinstance(answer, str) or not answer.strip():
            error = ProviderError("chat provider returned an empty response")
            self._emit_failure(run_id, "generation", generation_started, error)
            self._emit_failure(run_id, "answer", answer_started, error)
            raise error
        self._emit(
            run_id,
            "generation",
            "success",
            generation_started,
            {"model": self.chat_provider.model_name},
        )
        citations = build_citations(context_sources)
        text = answer
        if self.config.include_sources:
            text = append_sources(text, context_sources)
        if self.config.include_citations:
            text = append_citations(text, citations)
        result = RAGAnswer(
            text=text,
            sources=sources,
            context_sources=context_sources,
            citations=citations,
        )
        self._emit(
            run_id,
            "answer",
            "success",
            answer_started,
            {"source_count": len(sources), "citation_count": len(citations)},
        )
        return result

    def _system_prompt(self) -> str:
        if self.system_prompt_suffix is None:
            return _SYSTEM_PROMPT
        return (
            f"{_SYSTEM_PROMPT}\n\nAdditional application instructions:\n{self.system_prompt_suffix}"
        )

    def clear(self) -> None:
        """Remove every indexed document from the configured vector store."""

        run_id = self._new_run_id()
        started = perf_counter()
        try:
            removed_document_count = self.document_count if run_id is not None else None
            self.vector_store.clear()
        except Exception as error:
            self._emit_failure(run_id, "clear", started, error)
            raise
        details = (
            {"removed_document_count": removed_document_count}
            if removed_document_count is not None
            else {}
        )
        self._emit(run_id, "clear", "success", started, self._store_details(details))

    def _rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        limit: int,
        *,
        run_id: str | None = None,
    ) -> tuple[SearchResult, ...]:
        if self.reranker is None:
            return tuple(candidates[:limit])
        started = perf_counter()
        try:
            scores = tuple(self.reranker.rerank(query, candidates))
        except Exception as error:
            self._emit_failure(run_id, "reranking", started, error)
            raise RerankerError("reranker failed to score retrieved candidates") from error
        if len(scores) != len(candidates):
            error = RerankerError(
                f"reranker returned {len(scores)} scores for {len(candidates)} candidates"
            )
            self._emit_failure(run_id, "reranking", started, error)
            raise error
        if any(type(score) not in (int, float) or not math.isfinite(score) for score in scores):
            error = RerankerError("reranker scores must be finite numbers")
            self._emit_failure(run_id, "reranking", started, error)
            raise error
        ranked = [
            replace(result, rerank_score=float(score))
            for result, score in zip(candidates, scores, strict=True)
        ]
        ranked.sort(key=lambda result: result.rerank_score, reverse=True)
        selected = tuple(ranked[:limit])
        self._emit(
            run_id,
            "reranking",
            "success",
            started,
            {"candidate_count": len(candidates), "result_count": len(selected)},
        )
        return selected

    def _insufficient_context_answer(
        self,
        sources: Sequence[SearchResult],
        run_id: str | None,
        started: float,
    ) -> RAGAnswer:
        self._emit(
            run_id,
            "answer",
            "skipped",
            started,
            {"reason": "insufficient_context", "source_count": len(sources)},
        )
        return RAGAnswer(
            text=self.config.insufficient_context_response,
            sources=tuple(sources),
        )

    def _build_context(self, sources: Sequence[SearchResult]) -> str:
        return build_context(sources, self.config.max_context_chars)[0]

    @staticmethod
    def _validate_observers(observers: Sequence[Observer] | None) -> tuple[Observer, ...]:
        if observers is None:
            return ()
        values = tuple(observers)
        if any(not callable(getattr(observer, "on_event", None)) for observer in values):
            raise ConfigurationError("observers must define on_event(event)")
        return values

    def _new_run_id(self) -> str | None:
        return str(uuid4()) if self._observers else None

    def _emit(
        self,
        run_id: str | None,
        operation: str,
        status: str,
        started: float,
        details: Mapping[str, Any],
        *,
        error_type: str | None = None,
    ) -> None:
        """Notify observers without allowing observer failures to affect RAG work."""
        if run_id is None:
            return
        event = RAGEvent(
            run_id=run_id,
            operation=operation,
            status=status,  # type: ignore[arg-type]
            duration_ms=(perf_counter() - started) * 1_000,
            details=details,
            error_type=error_type,
        )
        for observer in self._observers:
            try:
                observer.on_event(event)
            except Exception:
                LOGGER.debug("PyAIStack observer failed", exc_info=True)

    def _emit_failure(
        self,
        run_id: str | None,
        operation: str,
        started: float,
        error: Exception,
    ) -> None:
        self._emit(
            run_id,
            operation,
            "failure",
            started,
            {},
            error_type=type(error).__name__,
        )

    def _metadata_details(
        self,
        details: Mapping[str, Any],
        documents: Sequence[Document],
    ) -> dict[str, Any]:
        """Include only application-allowlisted, bounded scalar metadata values."""
        result = dict(details)
        keys = self.config.observability_metadata_keys
        if not keys:
            return result
        values_by_key: dict[str, list[str | int | float | bool | None]] = {}
        for key in keys:
            values: list[str | int | float | bool | None] = []
            for document in documents:
                value = document.metadata.get(key)
                parts = value if isinstance(value, list) else [value]
                for part in parts:
                    if type(part) in (str, int, float, bool) or part is None:
                        if isinstance(part, str) and len(part) > 128:
                            continue
                        if part not in values:
                            values.append(part)
                    if len(values) == 10:
                        break
                if len(values) == 10:
                    break
            if values:
                values_by_key[key] = values
        if values_by_key:
            result["metadata"] = values_by_key
        return result

    def _store_details(self, details: Mapping[str, Any]) -> dict[str, Any]:
        """Add a bounded, non-sensitive backend identifier when a store provides one."""

        result = dict(details)
        name = getattr(self.vector_store, "observability_name", None)
        if (
            isinstance(name, str)
            and 0 < len(name) <= 32
            and all(
                character.isascii() and (character.isalnum() or character in "_-")
                for character in name
            )
        ):
            result["backend"] = name
        return result
