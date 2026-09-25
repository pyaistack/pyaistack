from __future__ import annotations

from contextlib import contextmanager

import pytest

from pyaistack.exceptions import (
    VectorStoreConnectionError,
    VectorStoreConstraintError,
    VectorStoreError,
    VectorStoreSchemaError,
    VectorStoreTimeoutError,
)
from pyaistack.vectorstores import PgVectorSchema, PgVectorStore
from pyaistack.vectorstores._postgres_filters import compile_filter
from pyaistack.vectorstores._postgres_schema import index_name, quote_identifier


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"dsn": "postgresql://example", "connection_factory": lambda: object()},
        {"dsn": "postgresql://example", "dimensions": 0},
        {"dsn": "postgresql://example", "dimensions": True},
        {"dsn": "postgresql://example", "dimensions": 2, "search_mode": "invalid"},
    ],
)
def test_pgvector_rejects_invalid_configuration_before_connecting(kwargs) -> None:
    kwargs.setdefault("dimensions", 2)
    with pytest.raises(VectorStoreError):
        PgVectorStore(**kwargs)


def test_schema_mapping_validates_and_quotes_identifiers() -> None:
    schema = PgVectorSchema(schema_name="application_ai", table_name="knowledge_chunks")

    assert schema.qualified_table == '"application_ai"."knowledge_chunks"'
    assert quote_identifier("valid_name_2") == '"valid_name_2"'
    assert len(index_name("x" * 60, "metadata")) <= 63

    with pytest.raises(VectorStoreSchemaError):
        PgVectorSchema(table_name="documents; DROP TABLE users")
    with pytest.raises(VectorStoreSchemaError, match="unique"):
        PgVectorSchema(id_column="text", text_column="text")


def test_postgres_filter_keeps_values_out_of_sql() -> None:
    hostile = "finance') OR TRUE --"
    statement, parameters = compile_filter(
        '"metadata"',
        {"category": hostile, "tags": ["budget", "saving"]},
    )

    assert hostile not in statement
    assert "category" not in statement
    assert any(hostile in str(parameter) for parameter in parameters)
    assert all(value in str(parameters) for value in ("budget", "saving"))


@pytest.mark.parametrize(
    "metadata_filter,expected_parameter",
    [
        ({"category": "finance"}, '"finance"'),
        ({"category": ["finance", "policy"]}, '["finance","policy"]'),
        ({"category": None}, "null"),
        ({"category": []}, "[]"),
    ],
)
def test_postgres_filter_supports_scalar_and_list_contract(
    metadata_filter, expected_parameter
) -> None:
    statement, parameters = compile_filter('"metadata"', metadata_filter)

    assert statement != "TRUE"
    assert statement.count("%s") == len(parameters)
    assert expected_parameter in parameters


class FakeConnection:
    def __init__(self) -> None:
        self.statements: list[tuple[str, object]] = []
        self.commits = 0
        self.rollbacks = 0

    def execute(self, statement, parameters=None):
        self.statements.append((statement, parameters))
        return self

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class QueryResult:
    def __init__(self, rows) -> None:
        self.rows = rows

    def fetchall(self):
        return self.rows


class QueryConnection:
    def __init__(self, rows) -> None:
        self.rows = rows
        self.statement = ""
        self.parameters = []

    def execute(self, statement, parameters=None):
        self.statement = statement
        self.parameters = parameters
        return QueryResult(self.rows)


class InsertCursor:
    def __init__(self) -> None:
        self.batches = []

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass

    def executemany(self, statement, rows) -> None:
        self.batches.append((statement, rows))


class PsycopgShapedConnection:
    """Connection deliberately has no executemany method, matching psycopg 3."""

    def __init__(self) -> None:
        self.insert_cursor = InsertCursor()

    def cursor(self):
        return self.insert_cursor


def operation_store(connection_factory):
    store = object.__new__(PgVectorStore)
    store._closed = False
    store._dsn = None
    store._connection_factory = connection_factory
    store.statement_timeout_ms = 1_000
    store.search_mode = "approximate"
    store.hnsw_ef_search = 20
    return store


def query_store() -> PgVectorStore:
    store = object.__new__(PgVectorStore)
    store.schema = PgVectorSchema()
    store.collection = "support"
    store.dimensions = 2
    store.max_candidates = 1_000
    store.max_query_chars = 4_096
    return store


def test_vector_query_is_bounded_parameterized_and_returns_documents() -> None:
    connection = QueryConnection([("doc-1", "policy", {"category": "finance"}, 0.75)])
    store = query_store()

    results = store._vector_search(
        connection,
        [1.0, 0.0],
        top_k=3,
        min_score=0.25,
        metadata_filter={"category": "finance"},
    )

    assert results[0].document.id == "doc-1"
    assert results[0].score == 0.75
    assert connection.statement.count("%s") == len(connection.parameters)
    assert connection.parameters[-2:] == [3, 0.25]
    assert "finance" not in connection.statement


def test_keyword_query_is_bounded_and_parameterized() -> None:
    connection = QueryConnection([("doc-1", "policy", {}, 0.5)])
    store = query_store()

    results = store._keyword_search(
        connection,
        "FIN-042",
        [1.0, 0.0],
        limit=5,
        metadata_filter={"tags": []},
    )

    assert results[0].score == 0.5
    assert connection.statement.count("%s") == len(connection.parameters)
    assert connection.parameters[0] == "FIN-042"
    assert "FIN-042" not in connection.statement


def test_retrieval_limits_are_enforced_before_database_work() -> None:
    store = query_store()
    store.max_candidates = 2
    store.max_query_chars = 5

    with pytest.raises(VectorStoreError, match="top_k"):
        store.search([1, 0], top_k=3)
    with pytest.raises(VectorStoreError, match="query cannot exceed"):
        store.hybrid_search("too long", [1, 0], top_k=1, candidate_k=2)


def test_batched_inserts_use_a_psycopg_cursor() -> None:
    connection = PsycopgShapedConnection()
    store = query_store()
    store.batch_size = 2
    rows = [("one",), ("two",), ("three",)]

    store._insert_batches(connection, "INSERT INTO documents VALUES (%s)", rows)

    assert connection.insert_cursor.batches == [
        ("INSERT INTO documents VALUES (%s)", [("one",), ("two",)]),
        ("INSERT INTO documents VALUES (%s)", [("three",)]),
    ]


def test_connection_factory_transaction_ownership() -> None:
    connection = FakeConnection()
    store = operation_store(lambda: connection)

    with store._operation("test") as acquired:
        assert acquired is connection

    assert connection.commits == 1
    assert connection.rollbacks == 0


def test_connection_factory_context_manager_and_rollback() -> None:
    connection = FakeConnection()
    returned = False

    @contextmanager
    def checkout():
        nonlocal returned
        try:
            yield connection
        finally:
            returned = True

    store = operation_store(checkout)

    with pytest.raises(VectorStoreError, match="RuntimeError") as captured:
        with store._operation("test"):
            raise RuntimeError("failure")

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert returned is True


class DatabaseFailure(Exception):
    def __init__(self, sqlstate: str) -> None:
        self.sqlstate = sqlstate


@pytest.mark.parametrize(
    "sqlstate,error_type",
    [
        ("57014", VectorStoreTimeoutError),
        ("08006", VectorStoreConnectionError),
        ("23505", VectorStoreConstraintError),
        ("XX000", VectorStoreError),
    ],
)
def test_database_errors_are_stable_and_sanitized(sqlstate, error_type) -> None:
    error = PgVectorStore._database_error("search", DatabaseFailure(sqlstate))

    assert isinstance(error, error_type)
    assert sqlstate not in str(error)


def test_closed_store_rejects_connection_factory_without_calling_it() -> None:
    called = False

    def factory():
        nonlocal called
        called = True
        return FakeConnection()

    store = operation_store(factory)
    store.close()

    with pytest.raises(VectorStoreError, match="closed"):
        with store._operation("test"):
            pass
    assert called is False
