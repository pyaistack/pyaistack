# Changelog

All notable changes to PyAIStack are documented in this file.

## [0.2.5]

### Added

- Optional PostgreSQL support through `pyaistack[postgres]` and psycopg 3.
- `PgVectorStore` with DSN or application-owned connection-factory modes.
- Managed, versioned pgvector schema creation and validated custom mappings
  through `PgVectorSchema`.
- Exact and HNSW cosine retrieval, PostgreSQL full-text hybrid retrieval,
  reciprocal-rank fusion, and parameterized JSONB metadata filters.
- Durable collection/model/dimension binding, atomic bounded-batch inserts,
  collection-scoped clearing, statement timeouts, and sanitized PostgreSQL
  exception types.
- PostgreSQL backend details in opt-in sanitized observability events.
- Unit coverage and an opt-in PostgreSQL integration suite.
- Separate PostgreSQL indexing, answer generation, retrieval inspection, and
  connection-factory examples.
  
## [0.2.4]

### Added

- Optional provider-neutral RAG observers with shared run IDs and sanitized completed-operation events.
- Versioned events with ISO 8601 UTC timestamps and instrumented `RAG.clear()` operations.
- Standard-library `InMemoryObserver`, `LoggingObserver`, and application-owned `JsonlObserver` implementations.
- Bounded, explicitly allowlisted indexing metadata in observability events; prompts, answers, document text, and credentials remain excluded by default.
- Retrieval evaluation reports with per-query returned IDs, first relevant rank, elapsed time, and JSON-compatible serialization.
- Observability guide and runnable example.

## [0.2.3]

### Added

- Pluggable `Reranker` support with bounded candidate retrieval and validated reranker scores.
- `SearchResult.rerank_score`, structured `Citation` objects, and verified context citations in `RAGAnswer`.
- `include_citations`, `min_context_results`, `rerank_candidate_k`, and configurable insufficient-context responses in `RAGConfig`.
- Dependency-free retrieval evaluation with labeled cases, Recall@k, and mean reciprocal rank (MRR).
- Dedicated reranking, citation, and evaluation examples.

### Changed

- Reorganized examples into basic, persistence, metadata, hybrid retrieval, and reranking learning paths.
- Separated SQLite and hybrid index creation from querying, removed machine-specific example paths, and added a bundled CSV metadata manifest.

## [0.2.2]

### Added

- Opt-in SQLite FTS5 + vector retrieval with reciprocal rank fusion.
- `retrieval_mode`, `candidate_k`, fusion scores and exact `context_sources`.
- Atomic keyword-index backfill and synchronization without re-embedding.
- Portable indexing/query demos and a reproducible retrieval benchmark.
- SQLite context-manager cleanup and optional store capability protocols.

### Fixed

- Context budgets include separators; displayed sources reflect supplied excerpts.
- SQLite filters before returning vector blobs and retains bounded top-k candidates.
- Configuration types, malformed vectors, metadata manifests and operational errors.
- Unicode, dates and version strings are preserved during metadata normalization.
- Explicit provider injection respects false-valued objects.
- Model binding is checked after clear/reuse and before indexing.

### Compatibility

- Vector mode remains default; hybrid is opt-in and requires SQLite FTS5.
- `score` remains cosine; `ranking_score` is the separate fusion score.
- Provider error causes are retained for debugging; normal formatting stays concise.
- No automatic rewriting of existing metadata or duplicate/resume handling.

## [0.2.1]

### Added

- UTF-8 `TextLoader` and recursive `DirectoryLoader` with `metadata_factory` support and required `file_type="text"`.
- Separator-aware `TextChunker` with overlap and one-based chunk metadata.
- Persistent `SQLiteVectorStore` for local exact cosine retrieval.
- Metadata filtering for `RAG.search()` and `RAG.ask()`.
- `FolderMetadataFactory` for deterministic metadata from folder structure.
- `CSVMetadataFactory` for metadata from a root-relative CSV manifest.
- `LLMMetadataFactory` using a provider-neutral chat interface and normalized list metadata.
- Sequential `DirectoryLoader.iter_load()` for one-file-at-a-time ingestion, but no detailed metadata.
- Provider protocols and explicit Ollama provider injection.
- Configurable source display, system-prompt suffixes, and package-specific errors.
- Dedicated documentations available. 

### Changed

- LLM-generated metadata is normalized into filter-friendly lowercase lists.
- Metadata filters match scalar fields and values contained in metadata lists.

## [0.1.1]

### Added

- Initial lightweight RAG runtime.
- Local Ollama embeddings and chat using `embeddinggemma` and `gemma3:4b` defaults.
- Thread-safe in-memory cosine-similarity vector store.
- Grounded answer generation with retrieved source results and scores.
