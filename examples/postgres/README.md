# PostgreSQL + pgvector examples

These examples use the local PostgreSQL connection below by default:

| Setting | Value |
| --- | --- |
| Host | `localhost` |
| Port | `5432` |
| User | `admin_user` |
| Database | `test_pyaistack` |

The password is intentionally not stored in the repository. Set the supplied
password for the current terminal through libpq's standard environment variable:

```bash
export PGPASSWORD="your-postgresql-password"
```

Install the optional PostgreSQL dependency and ensure the database already has
the pgvector extension enabled:

```bash
pip install -e ".[postgres]"
psql -h localhost -p 5432 -U admin_user -d test_pyaistack \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Start Ollama and pull the default embedding and chat models. Then index the
bundled text files once and query the persistent collection:

```bash
python examples/postgres/index.py
python examples/postgres/query.py "What does policy FIN-042 require?"
```

Use `search.py` when you want retrieval results, scores, metadata, and text
without asking the chat model to generate an answer:

```bash
python examples/postgres/search.py "FIN-042 budget policy" --top-k 5
```

`rag.search()` returns `SearchResult` objects. It does not return an answer with
an `.text` attribute; use `rag.ask()` for generated answers.

The direct examples let `PgVectorStore` open one connection per operation. The
factory example shows how an application can own connection acquisition and
cleanup instead:

```bash
python examples/postgres/connection_factory.py "How are backups handled?"
```

Use the custom-schema example to map PyAIStack fields to application-selected
schema, table, and column names:

```bash
python examples/postgres/custom_schema.py
```

For a self-contained demonstration it creates the mapped objects with
`initialize=True`. Production applications should normally apply reviewed
database migrations and reopen the compatible schema with `initialize=False`.

Override the defaults without changing source code:

```bash
export PYAISTACK_POSTGRES_DSN="host=db.example.com port=5432 dbname=app user=app_user sslmode=require"
export PYAISTACK_EMBEDDING_DIMENSIONS="768"
```

The configured dimension must match the embedding model and the PostgreSQL
`vector(n)` column. The default `embeddinggemma` configuration uses `768` here.
