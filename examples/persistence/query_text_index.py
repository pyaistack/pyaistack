"""Ask an existing SQLite index created by index_text_directory.py."""

import argparse

from pyaistack import RAG, RAGConfig
from pyaistack.vectorstores import SQLiteVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="What does policy FIN-042 require?")
    parser.add_argument("--db", default="text_knowledge.db")
    args = parser.parse_args()

    with SQLiteVectorStore(args.db) as store:
        rag = RAG(
            vector_store=store,
            config=RAGConfig(top_k=3, min_score=0.15, include_citations=True),
        )
        print(rag.ask(args.question).text)


if __name__ == "__main__":
    main()
