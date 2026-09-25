"""Query the existing PostgreSQL collection with hybrid retrieval."""

import argparse

from postgres_config import embedding_dimensions, postgres_dsn

from pyaistack import RAG, RAGConfig
from pyaistack.vectorstores import PgVectorStore

COLLECTION = "hybrid_knowledge"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="What does policy FIN-042 require?")
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
                top_k=5,
                candidate_k=20,
                min_score=0.25,
                include_citations=True,
            ),
        )
        print(rag.ask(args.question).text)


if __name__ == "__main__":
    main()
