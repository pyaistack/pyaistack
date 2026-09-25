"""First, Index the bundled text directory into a SQLite database once."""

import argparse
from pathlib import Path

from pyaistack import RAG
from pyaistack.chunking import TextChunker
from pyaistack.loaders import DirectoryLoader
from pyaistack.vectorstores import SQLiteVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="text_knowledge.db")
    parser.add_argument(
        "--directory", type=Path, default=Path(__file__).parents[1] / "hybrid_knowledge"
    )
    args = parser.parse_args()

    # Indexing and asking are separate: use query_text_index.py after this finishes.
    with SQLiteVectorStore(args.db) as store:
        if store:
            parser.error("index is already populated; use a new --db path to index again")
        rag = RAG(
            vector_store=store,
            chunker=TextChunker(chunk_size=800, chunk_overlap=100),
        )
        for document in DirectoryLoader(args.directory, file_type="text").iter_load():
            rag.add_documents([document])
        print(f"Indexed {rag.document_count} chunks in {args.db}")


if __name__ == "__main__":
    main()
