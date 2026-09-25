"""Query an existing SQLite index using FTS5 plus vector retrieval."""

import argparse

from pyaistack import RAG, RAGConfig
from pyaistack.vectorstores import SQLiteVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="What does policy FIN-02 require?")
    parser.add_argument("--db", default="hybrid_knowledge.db")
    args = parser.parse_args()
    with SQLiteVectorStore(args.db) as store:
        rag = RAG(
            vector_store=store,
            config=RAGConfig(
                retrieval_mode="hybrid",
                top_k=5,
                include_citations=True,
                min_score=0.15,
            ),
        )
        answer = rag.ask(args.question)
        print(answer.text)


if __name__ == "__main__":
    main()
