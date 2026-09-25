"""Query PostgreSQL through an application-owned connection factory."""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from postgres_config import embedding_dimensions, postgres_dsn

from pyaistack import RAG
from pyaistack.vectorstores import PgVectorStore

COLLECTION = "hybrid_knowledge"


@contextmanager
def connection_factory() -> Iterator[psycopg.Connection]:
    """Acquire and release one connection per PyAIStack operation.

    A production application can replace this function with a checkout from its
    existing connection pool without changing PgVectorStore or RAG usage.
    """

    connection = psycopg.connect(postgres_dsn(), connect_timeout=5)
    try:
        yield connection
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="How are backups handled?")
    args = parser.parse_args()

    with PgVectorStore(
        connection_factory=connection_factory,
        dimensions=embedding_dimensions(),
        collection=COLLECTION,
    ) as store:
        rag = RAG(vector_store=store)
        print(rag.ask(args.question).text)


if __name__ == "__main__":
    main()
