"""Inspect PostgreSQL retrieval results without generating an LLM answer."""

import argparse

from postgres_config import embedding_dimensions, postgres_dsn

from pyaistack import RAG, RAGConfig
from pyaistack.vectorstores import PgVectorStore

COLLECTION = "hybrid_knowledge"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="FIN-042 budget policy")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    with PgVectorStore(
        dsn=postgres_dsn(),
        dimensions=embedding_dimensions(),
        collection=COLLECTION,
    ) as store:
        rag = RAG(
            vector_store=store,
            config=RAGConfig(
                retrieval_mode="hybrid",
                top_k=args.top_k,
                candidate_k=max(20, args.top_k),
            ),
        )
        results = rag.search(args.query)

        for rank, result in enumerate(results, start=1):
            source = result.document.metadata.get("source", "unknown")
            chunk_index = result.document.metadata.get("chunk_index", "unknown")
            ranking_score = (
                f"{result.ranking_score:.4f}"
                if result.ranking_score is not None
                else "not applicable"
            )
            print(f"Result {rank}")
            print(f"Similarity score: {result.score:.4f}")
            print(f"Hybrid ranking score: {ranking_score}")
            print(f"Source: {source}")
            print(f"Chunk index: {chunk_index}")
            print(f"Text: {result.document.text}\n")


if __name__ == "__main__":
    main()
