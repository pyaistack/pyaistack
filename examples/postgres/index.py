"""Create a managed pgvector schema and index the bundled text directory."""

from pathlib import Path

from postgres_config import embedding_dimensions, postgres_dsn

from pyaistack import RAG
from pyaistack.chunking import TextChunker
from pyaistack.loaders import DirectoryLoader
from pyaistack.vectorstores import PgVectorStore

KNOWLEDGE_DIRECTORY = Path(__file__).parents[1] / "hybrid_knowledge"
COLLECTION = "hybrid_knowledge"


def main() -> None:
    with PgVectorStore(
        dsn=postgres_dsn(),
        dimensions=embedding_dimensions(),
        collection=COLLECTION,
        initialize=True,
    ) as store:
        if len(store):
            raise SystemExit(
                f"Collection {COLLECTION!r} already contains documents; run query.py instead."
            )

        rag = RAG(
            vector_store=store,
            chunker=TextChunker(chunk_size=800, chunk_overlap=100),
        )
        loader = DirectoryLoader(KNOWLEDGE_DIRECTORY, file_type="text")

        # Process one file at a time while PostgreSQL stores each resulting chunk.
        for document in loader.iter_load():
            rag.add_documents([document])

        print(f"Indexed {rag.document_count} chunks in collection {COLLECTION!r}.")


if __name__ == "__main__":
    main()
