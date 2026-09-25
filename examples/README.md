# PyAIStack examples

Run these examples from the repository root. Start Ollama and pull the default
models before running an example:

```bash
ollama pull embeddinggemma
ollama pull gemma3:4b
```

## Recommended sequence

1. Start with the in-memory RAG example.
2. Create a persistent SQLite index, then query it.
3. Choose a metadata strategy when your documents need filtering.
4. Try hybrid retrieval for keyword plus vector search.
5. Try reranking, citations, and retrieval evaluation.

## Basic RAG

| Example | Purpose | Command |
| --- | --- | --- |
| `examples/basic/basic_rag.py` | Smallest in-memory RAG flow. | `python examples/basic/basic_rag.py` |
| `examples/basic/chat.py` | Interactive in-memory chat with concise provider errors. | `python examples/basic/chat.py` |
| `examples/basic/custom_providers.py` | Explicitly select Ollama embedding and chat providers with model names. | `python examples/basic/custom_providers.py` |

## Persistent SQLite RAG

Index first, then use either query example against the same database.

```bash
python examples/persistence/index_text_directory.py
python examples/persistence/query_text_index.py
python examples/persistence/query_with_providers.py
```

`index_text_directory.py` uses the bundled `examples/hybrid_knowledge/` corpus
by default. Pass `--directory /path/to/text-files` and `--db my_index.db` to
use your own text directory and database path.

## PostgreSQL + pgvector RAG

The PostgreSQL examples use a managed pgvector schema and persistent collection.
Index the bundled directory first, then query it through either a direct DSN or
an application-owned connection factory.

| Example | Purpose | Command |
| --- | --- | --- |
| `examples/postgres/index.py` | Create the schema and index the bundled text directory. | `python examples/postgres/index.py` |
| `examples/postgres/query.py` | Reopen the collection and use hybrid retrieval. | `python examples/postgres/query.py` |
| `examples/postgres/search.py` | Inspect retrieved documents, scores, and metadata without answer generation. | `python examples/postgres/search.py "FIN-042 budget policy"` |
| `examples/postgres/connection_factory.py` | Supply application-owned connection lifecycle management. | `python examples/postgres/connection_factory.py` |
| `examples/postgres/custom_schema.py` | Map PyAIStack to application-selected PostgreSQL object and column names. | `python examples/postgres/custom_schema.py` |

See [`examples/postgres/README.md`](postgres/README.md) for local connection,
pgvector extension, password, and optional dependency setup.

## Metadata indexing

Each script indexes the bundled corpus into a separate SQLite database. They
are indexing examples; use a persistent query example afterwards.

| Example | Metadata source | Command |
| --- | --- | --- |
| `examples/metadata/index_single_file_metadata.py` | Fields supplied directly in application code. | `python examples/metadata/index_single_file_metadata.py` |
| `examples/metadata/index_folder_metadata.py` | Folder names and file names. | `python examples/metadata/index_folder_metadata.py` |
| `examples/metadata/index_csv_metadata.py` | Root-relative paths and JSON metadata in `metadata.csv`. | `python examples/metadata/index_csv_metadata.py` |
| `examples/metadata/index_llm_metadata.py` | A bounded file sample classified by the configured chat provider. | `python examples/metadata/index_llm_metadata.py` |

The CSV manifest contains one row for every text file in its selected directory.
The LLM metadata example makes one additional chat-model request per file.

## Hybrid retrieval

Hybrid retrieval combines SQLite FTS5 keyword matches with vector retrieval.
Index first, then query the saved database.

```bash
python examples/hybrid_retrieval/index.py
python examples/hybrid_retrieval/query.py "What does policy FIN-042 require?"
```

See [hybrid retrieval](../rag/hybrid-retrieval.md) for ranking and migration
behavior.

## Reranking, citations, and evaluation

| Example | Purpose | Command |
| --- | --- | --- |
| `examples/reranking/in_memory_evaluation.py` | Custom reranking, verified citations, Recall@k, and MRR with in-memory data. | `python examples/reranking/in_memory_evaluation.py` |
| `examples/reranking/hybrid_directory.py` | Directory indexing, hybrid retrieval, custom reranking, and verified citations. | `python examples/reranking/hybrid_directory.py` |

The included `KeywordReranker` is deliberately simple and dependency-free. For
a real application, replace it with a local cross-encoder or hosted reranking
provider as described in [reranking and verified citations](../rag/reranking.md).

## Observability

| Example | Purpose | Command |
| --- | --- | --- |
| `examples/observability/rag_events.py` | Opt in to sanitized standard logging and application-owned JSONL event storage. | `python examples/observability/rag_events.py` |

No events are created unless an observer is passed to `RAG`. See
[`docs/docs_pyaistack/rag/observability.md`](../docs/docs_pyaistack/rag/observability.md)
for event privacy and retrieval evaluation reports.
