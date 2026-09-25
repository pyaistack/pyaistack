"""Index a text directory with LLM-generated metadata for each file."""

import argparse
from pathlib import Path

from pyaistack import RAG
from pyaistack.chunking import TextChunker
from pyaistack.loaders import DirectoryLoader, LLMMetadataFactory
from pyaistack.providers import OllamaChatProvider
from pyaistack.vectorstores import SQLiteVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="llm_metadata.db")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument(
        "--directory", type=Path, default=Path(__file__).parents[1] / "hybrid_knowledge"
    )
    args = parser.parse_args()

    chat_provider = OllamaChatProvider(model="gemma3:4b", host=args.host)
    metadata_factory = LLMMetadataFactory(
        chat_provider,
        fields=("category", "topic", "document_type", "audience", "language", "tags"),
        max_sentences=8,
    )
    with SQLiteVectorStore(args.db) as store:
        if store:
            parser.error("index is already populated; use a new --db path to index again")
        rag = RAG(
            chat_provider=chat_provider,
            vector_store=store,
            chunker=TextChunker(chunk_size=800, chunk_overlap=100),
        )
        loader = DirectoryLoader(
            args.directory, file_type="text", metadata_factory=metadata_factory
        )
        # One file is classified, chunked, embedded, and stored before the next file starts.
        for document in loader.iter_load():
            rag.add_documents([document])
        print(f"Indexed {rag.document_count} chunks with LLM metadata in {args.db}")


if __name__ == "__main__":
    main()
