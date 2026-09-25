"""Map PyAIStack to custom PostgreSQL schema, table, and column names."""

from postgres_config import embedding_dimensions, postgres_dsn

from pyaistack import RAG
from pyaistack.vectorstores import PgVectorSchema, PgVectorStore

COLLECTION = "custom_schema_demo"

# Map PyAIStack's required fields to names selected by the application.
CUSTOM_SCHEMA = PgVectorSchema(
    schema_name="application_ai",
    table_name="knowledge_chunks",
    collection_table_name="knowledge_collections",
    collection_column="collection_key",
    id_column="chunk_id",
    text_column="content",
    metadata_column="attributes",
    vector_column="embedding_vector",
    created_at_column="created_at",
    updated_at_column="updated_at",
    collection_created_at_column="created_at",
    model_column="embedding_model",
    dimensions_column="embedding_dimensions",
    schema_version_column="adapter_schema_version",
)


def main() -> None:
    with PgVectorStore(
        dsn=postgres_dsn(),
        dimensions=embedding_dimensions(),
        collection=COLLECTION,
        schema=CUSTOM_SCHEMA,
        # This makes the example self-contained by creating the mapped objects.
        # In production, apply reviewed migrations and normally use initialize=False.
        initialize=True,
    ) as store:
        rag = RAG(vector_store=store)

        # Reuse existing rows when the example is run more than once.
        if rag.document_count == 0:
            rag.add(
                [
                    "PostgreSQL stores application data in tables and schemas.",
                    "pgvector adds vector columns and similarity operators to PostgreSQL.",
                ],
                metadatas=[
                    {"category": "database", "source": "postgresql"},
                    {"category": "database", "source": "pgvector"},
                ],
            )

        results = rag.search(
            "How are vectors stored in PostgreSQL?",
            top_k=2,
            metadata_filter={"category": "database"},
        )
        for result in results:
            print(f"Score: {result.score:.4f}")
            print(f"Source: {result.document.metadata.get('source', 'unknown')}")
            print(f"Text: {result.document.text}\n")


if __name__ == "__main__":
    main()
