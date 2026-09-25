"""Index the bundled text directory once for hybrid retrieval."""

import argparse
from pathlib import Path

from pyaistack import RAG
from pyaistack.chunking import TextChunker
from pyaistack.loaders import DirectoryLoader
from pyaistack.vectorstores import SQLiteVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="hybrid_knowledge.db")
    args = parser.parse_args()
    with SQLiteVectorStore(args.db) as store:
        if len(store):
            parser.error("index is already populated; query it or choose a new --db path")
        rag = RAG(vector_store=store, chunker=TextChunker(chunk_size=800, chunk_overlap=100))
        loader = DirectoryLoader(Path(__file__).parents[1] / "hybrid_knowledge", file_type="text")
        # Store the directory one file at a time to keep ingestion memory bounded.
        for document in loader.iter_load():
            rag.add_documents([document])
        print(f"Indexed {rag.document_count} chunks in {args.db}")


if __name__ == "__main__":
    main()
