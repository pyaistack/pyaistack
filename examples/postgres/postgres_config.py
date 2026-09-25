"""Shared local PostgreSQL settings for the pgvector examples."""

import os

# libpq reads the password separately from PGPASSWORD, so it is never stored in
# this repository. Override the complete connection string when required.
DEFAULT_CONNINFO = (
    "host=localhost port=5432 dbname=test_pyaistack user=admin_user"
)


def postgres_dsn() -> str:
    return os.environ.get("PYAISTACK_POSTGRES_DSN", DEFAULT_CONNINFO)


def embedding_dimensions() -> int:
    return int(os.environ.get("PYAISTACK_EMBEDDING_DIMENSIONS", "768"))
