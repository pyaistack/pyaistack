"""PostgreSQL schema mapping and identifier validation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, fields

from ..exceptions import VectorStoreSchemaError

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_identifier(value: str) -> str:
    """Validate and quote one simple PostgreSQL identifier."""

    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise VectorStoreSchemaError(
            "PostgreSQL identifiers must start with a letter or underscore and contain only "
            "letters, numbers, and underscores"
        )
    if len(value.encode("utf-8")) > 63:
        raise VectorStoreSchemaError("PostgreSQL identifiers cannot exceed 63 bytes")
    return f'"{value}"'


def index_name(table_name: str, purpose: str) -> str:
    """Create a stable PostgreSQL-length-safe index name."""

    raw = f"{table_name}_{purpose}_idx"
    if len(raw.encode("utf-8")) <= 63:
        return raw
    suffix = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"{raw[:54]}_{suffix}"


@dataclass(frozen=True, slots=True)
class PgVectorSchema:
    """Maps PyAIStack's required fields to a compatible PostgreSQL schema."""

    schema_name: str = "pyaistack"
    table_name: str = "documents"
    collection_table_name: str = "collections"
    collection_column: str = "collection_name"
    id_column: str = "document_id"
    text_column: str = "text"
    metadata_column: str = "metadata"
    vector_column: str = "embedding"
    created_at_column: str | None = "created_at"
    updated_at_column: str | None = "updated_at"
    collection_created_at_column: str | None = "created_at"
    model_column: str = "embedding_model"
    dimensions_column: str = "dimensions"
    schema_version_column: str = "schema_version"

    def __post_init__(self) -> None:
        values = [getattr(self, field.name) for field in fields(self)]
        for value in values:
            if value is not None:
                quote_identifier(value)
        document_columns = (
            self.collection_column,
            self.id_column,
            self.text_column,
            self.metadata_column,
            self.vector_column,
            *(column for column in (self.created_at_column, self.updated_at_column) if column),
        )
        if len(set(document_columns)) != len(document_columns):
            raise VectorStoreSchemaError("document column mappings must be unique")
        binding_columns = (
            self.collection_column,
            self.model_column,
            self.dimensions_column,
            self.schema_version_column,
            *(
                [self.collection_created_at_column]
                if self.collection_created_at_column is not None
                else []
            ),
        )
        if len(set(binding_columns)) != len(binding_columns):
            raise VectorStoreSchemaError("collection column mappings must be unique")

    def q(self, identifier: str) -> str:
        return quote_identifier(identifier)

    @property
    def qualified_table(self) -> str:
        return f"{self.q(self.schema_name)}.{self.q(self.table_name)}"

    @property
    def qualified_collection_table(self) -> str:
        return f"{self.q(self.schema_name)}.{self.q(self.collection_table_name)}"
