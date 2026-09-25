"""Index one text file with application metadata in SQLite."""

import argparse
from pathlib import Path

from pyaistack import RAG
from pyaistack.chunking import TextChunker
from pyaistack.loaders import LoadedDocument, TextLoader
from pyaistack.vectorstores import SQLiteVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="single_file_metadata.db")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(__file__).parents[1] / "hybrid_knowledge" / "finance-policy.txt",
    )
    args = parser.parse_args()

    # TextLoader supplies source. Add filterable application fields before indexing.
    loaded = TextLoader(args.file).load()[0]
    document = LoadedDocument(
        text=loaded.text,
        metadata={
            **loaded.metadata,
            "category": "finance",
            "topic": "purchase_approvals",
            "audience": "employees",
            "tags": ["policy", "fin_042"],
        },
    )
    with SQLiteVectorStore(args.db) as store:
        if store:
            parser.error("index is already populated; use a new --db path to index again")
        rag = RAG(vector_store=store, chunker=TextChunker(chunk_size=800, chunk_overlap=100))
        rag.add_documents([document])
        print(f"Indexed {rag.document_count} chunks with metadata in {args.db}")


if __name__ == "__main__":
    main()
