"""PostgreSQL + pgvector storage with collection-scoped retrieval."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any, Literal

from ..exceptions import (
    DependencyError,
    VectorStoreConnectionError,
    VectorStoreConstraintError,
    VectorStoreError,
    VectorStoreSchemaError,
    VectorStoreTimeoutError,
)
from ..rag.types import Document, SearchResult
from ._hybrid import fuse
from ._postgres_filters import compile_filter
from ._postgres_schema import PgVectorSchema, index_name, quote_identifier
from ._vectors import normalize, validate_search

SCHEMA_VERSION = 1
ConnectionFactory = Callable[[], Any]
SearchMode = Literal["exact", "approximate"]


def _load_psycopg() -> Any:
    try:
        return importlib.import_module("psycopg")
    except ImportError as error:
        raise DependencyError(
            'PostgreSQL support requires the optional dependency; install "pyaistack[postgres]"'
        ) from error


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in value.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(format(value, ".17g") for value in vector) + "]"


class PgVectorStore:
    """Persistent PostgreSQL store using pgvector and PostgreSQL full-text search.

    A DSN makes the store own each operation's connection. A connection factory
    may instead return either an application-owned connection or a context
    manager (for example, a pool checkout). Bare factory connections are not
    closed by the store.
    """

    observability_name = "postgresql"

    def __init__(
        self,
        *,
        dimensions: int,
        dsn: str | None = None,
        connection_factory: ConnectionFactory | None = None,
        collection: str = "default",
        schema: PgVectorSchema | None = None,
        initialize: bool = False,
        search_mode: SearchMode = "approximate",
        create_hnsw_index: bool = True,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
        hnsw_ef_search: int = 40,
        batch_size: int = 500,
        max_candidates: int = 1_000,
        max_query_chars: int = 4_096,
        statement_timeout_ms: int = 30_000,
        connect_timeout_seconds: int = 5,
    ) -> None:
        if (dsn is None) == (connection_factory is None):
            raise VectorStoreError("provide exactly one of dsn or connection_factory")
        if dsn is not None and (not isinstance(dsn, str) or not dsn.strip()):
            raise VectorStoreError("dsn must be a non-empty string")
        if connection_factory is not None and not callable(connection_factory):
            raise VectorStoreError("connection_factory must be callable")
        if type(dimensions) is not int or dimensions <= 0 or dimensions > 16_000:
            raise VectorStoreError("dimensions must be an integer between 1 and 16000")
        if not isinstance(collection, str) or not collection.strip():
            raise VectorStoreError("collection must be a non-empty string")
        if len(collection) > 255:
            raise VectorStoreError("collection cannot exceed 255 characters")
        if search_mode not in ("exact", "approximate"):
            raise VectorStoreError("search_mode must be 'exact' or 'approximate'")
        for name, value in (
            ("hnsw_m", hnsw_m),
            ("hnsw_ef_construction", hnsw_ef_construction),
            ("hnsw_ef_search", hnsw_ef_search),
            ("batch_size", batch_size),
            ("max_candidates", max_candidates),
            ("max_query_chars", max_query_chars),
            ("statement_timeout_ms", statement_timeout_ms),
            ("connect_timeout_seconds", connect_timeout_seconds),
        ):
            if type(value) is not int or value <= 0:
                raise VectorStoreError(f"{name} must be a positive integer")
        if not 2 <= hnsw_m <= 100:
            raise VectorStoreError("hnsw_m must be between 2 and 100")
        if not 4 <= hnsw_ef_construction <= 1_000:
            raise VectorStoreError("hnsw_ef_construction must be between 4 and 1000")
        if hnsw_ef_construction < 2 * hnsw_m:
            raise VectorStoreError("hnsw_ef_construction must be at least twice hnsw_m")
        if not 1 <= hnsw_ef_search <= 1_000:
            raise VectorStoreError("hnsw_ef_search must be between 1 and 1000")
        if create_hnsw_index and dimensions > 2_000:
            raise VectorStoreError("HNSW indexes support at most 2000 vector dimensions")

        self.dimensions = dimensions
        self.collection = collection.strip()
        self.schema = schema if schema is not None else PgVectorSchema()
        self.search_mode = search_mode
        self.create_hnsw_index = bool(create_hnsw_index)
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.batch_size = batch_size
        self.max_candidates = max_candidates
        self.max_query_chars = max_query_chars
        self.statement_timeout_ms = statement_timeout_ms
        self.connect_timeout_seconds = connect_timeout_seconds
        self._dsn = dsn.strip() if dsn is not None else None
        self._connection_factory = connection_factory
        self._closed = False
        self._model_name: str | None = None
        self._psycopg = _load_psycopg()

        if initialize:
            self._initialize_schema()
        self.validate_schema()
        with self._operation("register collection", write=True) as connection:
            self._ensure_collection(connection)

    @contextmanager
    def _raw_connection(self) -> Iterator[Any]:
        if self._closed:
            raise VectorStoreError("PostgreSQL store is closed")
        owned = self._dsn is not None
        resource: Any
        if owned:
            resource = self._psycopg.connect(
                self._dsn,
                connect_timeout=self.connect_timeout_seconds,
            )
        else:
            assert self._connection_factory is not None
            resource = self._connection_factory()
            if resource is None:
                raise VectorStoreConnectionError("connection_factory returned no connection")

        # psycopg connections are context managers too, but a bare connection
        # returned by an application factory remains application-owned.
        is_bare_connection = callable(getattr(resource, "execute", None))
        manager = nullcontext(resource) if is_bare_connection else resource
        if not is_bare_connection and not hasattr(resource, "__enter__"):
            raise VectorStoreConnectionError(
                "connection_factory must return a connection or connection context manager"
            )
        try:
            with manager as connection:
                if connection is None:
                    raise VectorStoreConnectionError(
                        "connection factory context manager returned no connection"
                    )
                yield connection
        finally:
            if owned and is_bare_connection:
                resource.close()

    @contextmanager
    def _operation(self, name: str, *, write: bool = False) -> Iterator[Any]:
        try:
            with self._raw_connection() as connection:
                self._configure_connection(connection)
                try:
                    yield connection
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise
        except VectorStoreError:
            raise
        except Exception as error:
            raise self._database_error(name, error) from error

    def _configure_connection(self, connection: Any) -> None:
        connection.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (str(self.statement_timeout_ms),),
        )
        if self.search_mode == "exact":
            connection.execute("SELECT set_config('enable_indexscan', 'off', true)")
            connection.execute("SELECT set_config('enable_bitmapscan', 'off', true)")
        else:
            connection.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)",
                (str(self.hnsw_ef_search),),
            )

    @staticmethod
    def _database_error(operation: str, error: Exception) -> VectorStoreError:
        sqlstate = getattr(error, "sqlstate", None)
        if sqlstate == "57014":
            return VectorStoreTimeoutError(f"PostgreSQL {operation} timed out")
        if (isinstance(sqlstate, str) and sqlstate.startswith("08")) or type(error).__name__ in {
            "OperationalError",
            "ConnectionTimeout",
        }:
            return VectorStoreConnectionError(f"PostgreSQL {operation} could not connect")
        if isinstance(sqlstate, str) and sqlstate.startswith("23"):
            return VectorStoreConstraintError(
                f"PostgreSQL {operation} violated a database constraint"
            )
        return VectorStoreError(
            f"PostgreSQL {operation} failed ({type(error).__name__})"
        )

    def _extension_version(self, connection: Any) -> str:
        row = connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        if row is None:
            raise VectorStoreSchemaError(
                "pgvector extension is not enabled; a database administrator must run "
                "CREATE EXTENSION vector"
            )
        return str(row[0])

    def _initialize_schema(self) -> None:
        schema = self.schema
        collection_created_at = (
            f", {schema.q(schema.collection_created_at_column)} "
            "TIMESTAMPTZ NOT NULL DEFAULT now()"
            if schema.collection_created_at_column is not None
            else ""
        )
        document_created_at = (
            f", {schema.q(schema.created_at_column)} TIMESTAMPTZ NOT NULL DEFAULT now()"
            if schema.created_at_column is not None
            else ""
        )
        document_updated_at = (
            f", {schema.q(schema.updated_at_column)} TIMESTAMPTZ NOT NULL DEFAULT now()"
            if schema.updated_at_column is not None
            else ""
        )
        with self._operation("schema initialization", write=True) as connection:
            extension_version = self._extension_version(connection)
            if self.create_hnsw_index and _version_tuple(extension_version) < (0, 5, 0):
                raise VectorStoreSchemaError("HNSW indexing requires pgvector 0.5.0 or newer")
            connection.execute(f"CREATE SCHEMA IF NOT EXISTS {schema.q(schema.schema_name)}")
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS {schema.qualified_collection_table} ("
                f"{schema.q(schema.collection_column)} TEXT PRIMARY KEY, "
                f"{schema.q(schema.model_column)} TEXT, "
                f"{schema.q(schema.dimensions_column)} INTEGER NOT NULL, "
                f"{schema.q(schema.schema_version_column)} INTEGER NOT NULL"
                f"{collection_created_at}, "
                f"CHECK ({schema.q(schema.dimensions_column)} > 0)"
                ")"
            )
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS {schema.qualified_table} ("
                f"{schema.q(schema.collection_column)} TEXT NOT NULL REFERENCES "
                f"{schema.qualified_collection_table} "
                f"({schema.q(schema.collection_column)}) ON DELETE CASCADE, "
                f"{schema.q(schema.id_column)} TEXT NOT NULL, "
                f"{schema.q(schema.text_column)} TEXT NOT NULL, "
                f"{schema.q(schema.metadata_column)} JSONB NOT NULL DEFAULT '{{}}'::jsonb, "
                f"{schema.q(schema.vector_column)} vector({self.dimensions}) NOT NULL"
                f"{document_created_at}{document_updated_at}, "
                f"PRIMARY KEY ({schema.q(schema.collection_column)}, "
                f"{schema.q(schema.id_column)})"
                ")"
            )
            self._create_indexes(connection)

    def _create_indexes(self, connection: Any) -> None:
        schema = self.schema
        collection_index = quote_identifier(index_name(schema.table_name, "collection"))
        metadata_index = quote_identifier(index_name(schema.table_name, "metadata"))
        text_index = quote_identifier(index_name(schema.table_name, "text_search"))
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS {collection_index} ON {schema.qualified_table} "
            f"({schema.q(schema.collection_column)})"
        )
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS {metadata_index} ON {schema.qualified_table} "
            f"USING gin ({schema.q(schema.metadata_column)})"
        )
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS {text_index} ON {schema.qualified_table} "
            f"USING gin (to_tsvector('simple', coalesce({schema.q(schema.text_column)}, '')))"
        )
        if self.create_hnsw_index:
            vector_index = quote_identifier(index_name(schema.table_name, "vector_hnsw"))
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS {vector_index} ON {schema.qualified_table} "
                f"USING hnsw ({schema.q(schema.vector_column)} vector_cosine_ops) "
                f"WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_ef_construction})"
            )

    def validate_schema(self) -> None:
        """Verify pgvector and every required mapped column without changing data."""

        with self._operation("schema validation") as connection:
            extension_version = self._extension_version(connection)
            if self.create_hnsw_index and _version_tuple(extension_version) < (0, 5, 0):
                raise VectorStoreSchemaError("HNSW indexing requires pgvector 0.5.0 or newer")
            document_columns = self._column_info(connection, self.schema.table_name)
            collection_columns = self._column_info(
                connection, self.schema.collection_table_name
            )
            expected_document = {
                self.schema.collection_column: "text",
                self.schema.id_column: "text",
                self.schema.text_column: "text",
                self.schema.metadata_column: "jsonb",
                self.schema.vector_column: f"vector({self.dimensions})",
            }
            if self.schema.created_at_column is not None:
                expected_document[self.schema.created_at_column] = "timestamp with time zone"
            if self.schema.updated_at_column is not None:
                expected_document[self.schema.updated_at_column] = "timestamp with time zone"
            expected_collection = {
                self.schema.collection_column: "text",
                self.schema.model_column: "text",
                self.schema.dimensions_column: "integer",
                self.schema.schema_version_column: "integer",
            }
            if self.schema.collection_created_at_column is not None:
                expected_collection[self.schema.collection_created_at_column] = (
                    "timestamp with time zone"
                )
            self._validate_column_types("document", document_columns, expected_document)
            self._validate_column_types("collection", collection_columns, expected_collection)
            self._validate_not_null(
                "document",
                document_columns,
                {
                    self.schema.collection_column,
                    self.schema.id_column,
                    self.schema.text_column,
                    self.schema.metadata_column,
                    self.schema.vector_column,
                },
            )
            self._validate_not_null(
                "collection",
                collection_columns,
                {
                    self.schema.collection_column,
                    self.schema.dimensions_column,
                    self.schema.schema_version_column,
                },
            )
            self._validate_unique_columns(
                connection,
                self.schema.collection_table_name,
                {self.schema.collection_column},
                "collection",
            )
            self._validate_unique_columns(
                connection,
                self.schema.table_name,
                {self.schema.collection_column, self.schema.id_column},
                "document",
            )
            if self.search_mode == "approximate":
                self._validate_ann_index(connection)

    def _column_info(self, connection: Any, table_name: str) -> dict[str, tuple[str, bool]]:
        rows = connection.execute(
            "SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull "
            "FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = %s AND c.relname = %s "
            "AND a.attnum > 0 AND NOT a.attisdropped",
            (self.schema.schema_name, table_name),
        ).fetchall()
        if not rows:
            raise VectorStoreSchemaError(
                f"PostgreSQL table {self.schema.schema_name}.{table_name} does not exist"
            )
        return {
            str(name): (str(data_type), bool(not_null))
            for name, data_type, not_null in rows
        }

    @staticmethod
    def _validate_column_types(
        table_role: str,
        actual: Mapping[str, tuple[str, bool]],
        expected: Mapping[str, str],
    ) -> None:
        missing = sorted(set(expected) - set(actual))
        if missing:
            raise VectorStoreSchemaError(
                f"PostgreSQL {table_role} table is missing required columns: "
                + ", ".join(missing)
            )
        compatible = {
            "text": {"text", "character varying"},
            "integer": {"smallint", "integer", "bigint"},
            "timestamp with time zone": {"timestamp with time zone"},
            "jsonb": {"jsonb"},
        }
        mismatches = []
        for name, data_type in expected.items():
            accepted = compatible.get(data_type, {data_type})
            actual_type = actual[name][0]
            if actual_type not in accepted:
                mismatches.append(f"{name} must be {data_type}, found {actual_type}")
        if mismatches:
            raise VectorStoreSchemaError(
                f"PostgreSQL {table_role} table has incompatible columns: "
                + "; ".join(mismatches)
            )

    @staticmethod
    def _validate_not_null(
        table_role: str,
        columns: Mapping[str, tuple[str, bool]],
        required: set[str],
    ) -> None:
        nullable = sorted(name for name in required if not columns[name][1])
        if nullable:
            raise VectorStoreSchemaError(
                f"PostgreSQL {table_role} table columns must be NOT NULL: "
                + ", ".join(nullable)
            )

    def _validate_unique_columns(
        self,
        connection: Any,
        table_name: str,
        required: set[str],
        table_role: str,
    ) -> None:
        rows = connection.execute(
            "SELECT array_agg(a.attname ORDER BY keys.ordinality) "
            "FROM pg_index i "
            "JOIN pg_class c ON c.oid = i.indrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "CROSS JOIN LATERAL unnest(i.indkey) WITH ORDINALITY keys(attnum, ordinality) "
            "JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = keys.attnum "
            "WHERE n.nspname = %s AND c.relname = %s AND i.indisunique "
            "AND i.indisvalid AND i.indpred IS NULL AND i.indexprs IS NULL "
            "AND keys.ordinality <= i.indnkeyatts "
            "GROUP BY i.indexrelid",
            (self.schema.schema_name, table_name),
        ).fetchall()
        if not any(len(names) == len(required) and set(names) == required for (names,) in rows):
            names = ", ".join(sorted(required))
            raise VectorStoreSchemaError(
                f"PostgreSQL {table_role} table requires a unique constraint or index on: "
                f"{names}"
            )

    def _validate_ann_index(self, connection: Any) -> None:
        row = connection.execute(
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_index i "
            "JOIN pg_class c ON c.oid = i.indrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "JOIN pg_class ic ON ic.oid = i.indexrelid "
            "JOIN pg_am am ON am.oid = ic.relam "
            "CROSS JOIN LATERAL unnest(i.indkey) WITH ORDINALITY keys(attnum, ordinality) "
            "CROSS JOIN LATERAL unnest(i.indclass) WITH ORDINALITY classes(opcoid, ordinality) "
            "JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = keys.attnum "
            "JOIN pg_opclass oc ON oc.oid = classes.opcoid "
            "WHERE n.nspname = %s AND c.relname = %s "
            "AND keys.ordinality = classes.ordinality "
            "AND a.attname = %s AND am.amname IN ('hnsw', 'ivfflat') "
            "AND oc.opcname = 'vector_cosine_ops' "
            "AND i.indisvalid"
            ")",
            (
                self.schema.schema_name,
                self.schema.table_name,
                self.schema.vector_column,
            ),
        ).fetchone()
        if row is None or not bool(row[0]):
            raise VectorStoreSchemaError(
                "approximate search requires a valid HNSW or IVFFlat index on the vector column; "
                "use search_mode='exact' for an unindexed compatible schema"
            )

    def health_check(self) -> None:
        """Verify connectivity and the mapped schema without writing documents."""

        self.validate_schema()
        with self._operation("health check") as connection:
            connection.execute("SELECT 1").fetchone()

    def _ensure_collection(self, connection: Any) -> tuple[str | None, int, int]:
        schema = self.schema
        connection.execute(
            f"INSERT INTO {schema.qualified_collection_table} "
            f"({schema.q(schema.collection_column)}, {schema.q(schema.model_column)}, "
            f"{schema.q(schema.dimensions_column)}, {schema.q(schema.schema_version_column)}) "
            "VALUES (%s, NULL, %s, %s) "
            f"ON CONFLICT ({schema.q(schema.collection_column)}) DO NOTHING",
            (self.collection, self.dimensions, SCHEMA_VERSION),
        )
        return self._read_binding(connection, for_update=True)

    def _read_binding(
        self, connection: Any, *, for_update: bool = False
    ) -> tuple[str | None, int, int]:
        schema = self.schema
        lock = " FOR UPDATE" if for_update else ""
        row = connection.execute(
            f"SELECT {schema.q(schema.model_column)}, "
            f"{schema.q(schema.dimensions_column)}, {schema.q(schema.schema_version_column)} "
            f"FROM {schema.qualified_collection_table} "
            f"WHERE {schema.q(schema.collection_column)} = %s{lock}",
            (self.collection,),
        ).fetchone()
        if row is None:
            raise VectorStoreSchemaError("PostgreSQL collection is not registered")
        model, dimensions, version = row
        if int(dimensions) != self.dimensions:
            raise VectorStoreError(
                f"collection dimension is {dimensions}, not configured dimension {self.dimensions}"
            )
        if int(version) != SCHEMA_VERSION:
            raise VectorStoreSchemaError(
                f"unsupported PostgreSQL schema version {version}; migration is required"
            )
        return (None if model is None else str(model), int(dimensions), int(version))

    def set_embedding_model(self, model_name: str) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise VectorStoreError("embedding model name cannot be empty")
        model_name = model_name.strip()
        schema = self.schema
        with self._operation("model binding", write=True) as connection:
            current, _, _ = self._ensure_collection(connection)
            if current is not None and current != model_name:
                raise VectorStoreError(
                    f"collection uses embedding model {current}, not {model_name}; "
                    "use a different collection or table"
                )
            if current is None:
                connection.execute(
                    f"UPDATE {schema.qualified_collection_table} "
                    f"SET {schema.q(schema.model_column)} = %s "
                    f"WHERE {schema.q(schema.collection_column)} = %s",
                    (model_name, self.collection),
                )
            self._model_name = model_name

    def _check_binding(self, connection: Any) -> None:
        current, _, _ = self._read_binding(connection)
        if self._model_name is not None and current != self._model_name:
            raise VectorStoreError(
                "embedding model binding changed; reopen with the correct model"
            )

    def add(self, documents: Sequence[Document], vectors: Sequence[Sequence[float]]) -> None:
        if len(documents) != len(vectors):
            raise VectorStoreError("documents and vectors must have the same length")
        if not documents:
            return
        normalized = [normalize(vector) for vector in vectors]
        if any(len(vector) != self.dimensions for vector in normalized):
            raise VectorStoreError(
                f"embedding vectors must have configured dimension {self.dimensions}"
            )

        rows: list[tuple[Any, ...]] = []
        for document, vector in zip(documents, normalized, strict=True):
            if not isinstance(document.id, str) or not document.id:
                raise VectorStoreError("document IDs must be non-empty strings")
            if not isinstance(document.text, str) or not document.text:
                raise VectorStoreError("document text must be a non-empty string")
            if any(not isinstance(key, str) for key in document.metadata):
                raise VectorStoreError("metadata keys must be strings")
            try:
                metadata = json.dumps(document.metadata, allow_nan=False, separators=(",", ":"))
            except (TypeError, ValueError) as error:
                raise VectorStoreError("metadata must be JSON-compatible") from error
            rows.append(
                (self.collection, document.id, document.text, metadata, _vector_literal(vector))
            )

        schema = self.schema
        statement = (
            f"INSERT INTO {schema.qualified_table} "
            f"({schema.q(schema.collection_column)}, {schema.q(schema.id_column)}, "
            f"{schema.q(schema.text_column)}, {schema.q(schema.metadata_column)}, "
            f"{schema.q(schema.vector_column)}) "
            "VALUES (%s, %s, %s, %s::jsonb, %s::vector)"
        )
        with self._operation("document insert", write=True) as connection:
            self._check_binding(connection)
            self._insert_batches(connection, statement, rows)

    def _insert_batches(
        self,
        connection: Any,
        statement: str,
        rows: Sequence[tuple[Any, ...]],
    ) -> None:
        """Insert bounded batches through a psycopg cursor."""

        with connection.cursor() as cursor:
            for offset in range(0, len(rows), self.batch_size):
                cursor.executemany(statement, rows[offset : offset + self.batch_size])

    def _validate_query(self, query_vector: Sequence[float]) -> tuple[float, ...]:
        vector = normalize(query_vector)
        if len(vector) != self.dimensions:
            raise VectorStoreError(
                f"query vector must have configured dimension {self.dimensions}"
            )
        return vector

    def _vector_search(
        self,
        connection: Any,
        query: Sequence[float],
        *,
        top_k: int,
        min_score: float | None,
        metadata_filter: Mapping[str, Any] | None,
    ) -> list[SearchResult]:
        schema = self.schema
        metadata_sql, filter_parameters = compile_filter(
            schema.q(schema.metadata_column), metadata_filter
        )
        literal = _vector_literal(query)
        score_filter = "" if min_score is None else "WHERE (1 - distance) >= %s "
        parameters: list[Any] = [literal, self.collection, *filter_parameters, literal, top_k]
        if min_score is not None:
            parameters.append(float(min_score))
        statement = (
            "WITH nearest AS ("
            f"SELECT {schema.q(schema.id_column)}, {schema.q(schema.text_column)}, "
            f"{schema.q(schema.metadata_column)}, "
            f"({schema.q(schema.vector_column)} <=> %s::vector) AS distance "
            f"FROM {schema.qualified_table} "
            f"WHERE {schema.q(schema.collection_column)} = %s AND ({metadata_sql}) "
            f"ORDER BY {schema.q(schema.vector_column)} <=> %s::vector "
            "LIMIT %s"
            ") "
            f"SELECT {schema.q(schema.id_column)}, {schema.q(schema.text_column)}, "
            f"{schema.q(schema.metadata_column)}, (1 - distance) AS score FROM nearest "
            f"{score_filter}"
            f"ORDER BY distance, {schema.q(schema.id_column)}"
        )
        rows = connection.execute(statement, parameters).fetchall()
        return [self._result(row) for row in rows]

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        min_score: float | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        validate_search(top_k, min_score)
        if top_k > self.max_candidates:
            raise VectorStoreError(f"top_k cannot exceed {self.max_candidates}")
        query = self._validate_query(query_vector)
        with self._operation("vector search") as connection:
            self._check_binding(connection)
            return self._vector_search(
                connection,
                query,
                top_k=top_k,
                min_score=min_score,
                metadata_filter=metadata_filter,
            )

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
        if top_k > self.max_candidates or candidate_k > self.max_candidates:
            raise VectorStoreError(
                f"top_k and candidate_k cannot exceed {self.max_candidates}"
            )
        if not isinstance(query, str) or not query.strip():
            raise VectorStoreError("query must be a non-empty string")
        if len(query) > self.max_query_chars:
            raise VectorStoreError(f"query cannot exceed {self.max_query_chars} characters")
        vector = self._validate_query(query_vector)
        limit = max(top_k, candidate_k)
        with self._operation("hybrid search") as connection:
            self._check_binding(connection)
            semantic = self._vector_search(
                connection,
                vector,
                top_k=limit,
                min_score=None,
                metadata_filter=metadata_filter,
            )
            lexical = self._keyword_search(
                connection,
                query.strip(),
                vector,
                limit=limit,
                metadata_filter=metadata_filter,
            )
            return fuse(semantic, lexical, top_k, min_score)

    def _keyword_search(
        self,
        connection: Any,
        query: str,
        vector: Sequence[float],
        *,
        limit: int,
        metadata_filter: Mapping[str, Any] | None,
    ) -> list[SearchResult]:
        schema = self.schema
        metadata_sql, filter_parameters = compile_filter(
            schema.q(schema.metadata_column), metadata_filter
        )
        text_vector = f"to_tsvector('simple', coalesce({schema.q(schema.text_column)}, ''))"
        statement = (
            "WITH search_query AS (SELECT plainto_tsquery('simple', %s) AS query), "
            "lexical AS ("
            f"SELECT {schema.q(schema.id_column)}, {schema.q(schema.text_column)}, "
            f"{schema.q(schema.metadata_column)}, {schema.q(schema.vector_column)}, "
            f"ts_rank_cd({text_vector}, search_query.query) AS text_rank "
            f"FROM {schema.qualified_table} CROSS JOIN search_query "
            f"WHERE {schema.q(schema.collection_column)} = %s AND ({metadata_sql}) "
            f"AND {text_vector} @@ search_query.query "
            f"ORDER BY text_rank DESC, {schema.q(schema.id_column)} LIMIT %s"
            ") "
            f"SELECT {schema.q(schema.id_column)}, {schema.q(schema.text_column)}, "
            f"{schema.q(schema.metadata_column)}, "
            f"(1 - ({schema.q(schema.vector_column)} <=> %s::vector)) AS score "
            f"FROM lexical ORDER BY text_rank DESC, {schema.q(schema.id_column)}"
        )
        parameters = [query, self.collection, *filter_parameters, limit, _vector_literal(vector)]
        rows = connection.execute(statement, parameters).fetchall()
        return [self._result(row) for row in rows]

    @staticmethod
    def _result(row: Sequence[Any]) -> SearchResult:
        metadata = row[2]
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError as error:
                raise VectorStoreError("stored PostgreSQL metadata is invalid JSON") from error
        if not isinstance(metadata, dict):
            raise VectorStoreError("stored PostgreSQL metadata must be an object")
        score = float(row[3])
        if not math.isfinite(score):
            raise VectorStoreError("stored PostgreSQL vector produced a non-finite score")
        return SearchResult(
            document=Document(id=str(row[0]), text=str(row[1]), metadata=metadata),
            score=max(-1.0, min(1.0, score)),
        )

    def clear(self) -> None:
        schema = self.schema
        with self._operation("collection clear", write=True) as connection:
            self._check_binding(connection)
            connection.execute(
                f"DELETE FROM {schema.qualified_table} "
                f"WHERE {schema.q(schema.collection_column)} = %s",
                (self.collection,),
            )

    def __len__(self) -> int:
        schema = self.schema
        with self._operation("document count") as connection:
            self._check_binding(connection)
            row = connection.execute(
                f"SELECT COUNT(*) FROM {schema.qualified_table} "
                f"WHERE {schema.q(schema.collection_column)} = %s",
                (self.collection,),
            ).fetchone()
            return int(row[0])

    def close(self) -> None:
        """Mark the store closed; application-owned pools remain application-owned."""

        self._closed = True

    def __enter__(self) -> PgVectorStore:
        if self._closed:
            raise VectorStoreError("PostgreSQL store is closed")
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


__all__ = ["PgVectorSchema", "PgVectorStore"]
