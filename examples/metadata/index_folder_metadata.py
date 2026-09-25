"""Index a text directory with deterministic metadata derived from paths."""

import argparse
from pathlib import Path

from pyaistack import RAG
from pyaistack.chunking import TextChunker
from pyaistack.loaders import DirectoryLoader, FolderMetadataFactory
from pyaistack.vectorstores import SQLiteVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="folder_metadata.db")
    parser.add_argument(
        "--directory", type=Path, default=Path(__file__).parents[1] / "hybrid_knowledge"
    )
    args = parser.parse_args()

    # Folder names map to category, topic, and so on when the directory is nested.
    loader = DirectoryLoader(
        args.directory,
        file_type="text",
        metadata_factory=FolderMetadataFactory(args.directory),
    )
    with SQLiteVectorStore(args.db) as store:
        if store:
            parser.error("index is already populated; use a new --db path to index again")
        rag = RAG(vector_store=store, chunker=TextChunker(chunk_size=800, chunk_overlap=100))
        for document in loader.iter_load():
            rag.add_documents([document])
        print(f"Indexed {rag.document_count} chunks with folder metadata in {args.db}")


if __name__ == "__main__":
    main()
