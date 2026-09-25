"""Index a text directory using metadata from a portable CSV manifest."""

import argparse
from pathlib import Path

from pyaistack import RAG
from pyaistack.chunking import TextChunker
from pyaistack.loaders import CSVMetadataFactory, DirectoryLoader
from pyaistack.vectorstores import SQLiteVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="csv_metadata.db")
    parser.add_argument(
        "--directory", type=Path, default=Path(__file__).parents[1] / "hybrid_knowledge"
    )
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("metadata.csv"))
    args = parser.parse_args()

    # Every .txt file under --directory needs one root-relative row in the manifest.
    metadata_factory = CSVMetadataFactory(args.manifest, root=args.directory)
    loader = DirectoryLoader(args.directory, file_type="text", metadata_factory=metadata_factory)
    with SQLiteVectorStore(args.db) as store:
        if store:
            parser.error("index is already populated; use a new --db path to index again")
        rag = RAG(vector_store=store, chunker=TextChunker(chunk_size=800, chunk_overlap=100))
        for document in loader.iter_load():
            rag.add_documents([document])
        print(f"Indexed {rag.document_count} chunks with CSV metadata in {args.db}")


if __name__ == "__main__":
    main()
