"""Index the bundled text directory and query it with hybrid retrieval and reranking.

Start Ollama and pull the default models first:
    ollama pull embeddinggemma
    ollama pull gemma3:4b

Then run:
    python examples/reranking/hybrid_directory.py

Optional custom query:
    python examples/reranking/hybrid_directory.py "What is the password reset process?"

The SQLite index is created only when the selected database is empty. Delete or
rename that local database yourself when you want to index the corpus again.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

from pyaistack import RAG, RAGConfig, SearchResult
from pyaistack.chunking import TextChunker
from pyaistack.loaders import DirectoryLoader
from pyaistack.vectorstores import SQLiteVectorStore


class KeywordReranker:
    """Transparent demonstration reranker; replace with a model-based reranker in production."""

    def rerank(self, query: str, candidates: Sequence[SearchResult]) -> Sequence[float]:
        query_terms = set(re.findall(r"\w+", query.lower()))
        return [
            float(len(query_terms.intersection(re.findall(r"\w+", item.document.text.lower()))))
            for item in candidates
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        default="What does policy FIN-042 require?",
    )
    parser.add_argument("--db", default="hybrid_rerank_knowledge.db")
    args = parser.parse_args()

    knowledge_directory = Path(__file__).parents[1] / "hybrid_knowledge"
    with SQLiteVectorStore(args.db) as store:
        rag = RAG(
            vector_store=store,
            chunker=TextChunker(chunk_size=600, chunk_overlap=100),
            reranker=KeywordReranker(),
            config=RAGConfig(
                retrieval_mode="hybrid",
                top_k=3,
                candidate_k=10,
                rerank_candidate_k=10,
                include_citations=True,
            ),
        )

        if not store:
            loader = DirectoryLoader(knowledge_directory, file_type="text")
            for document in loader.iter_load():
                rag.add_documents([document])
            print(f"Indexed {rag.document_count} chunks from {knowledge_directory}")

        answer = rag.ask(args.question)
        print(f"\n{answer.text}")


if __name__ == "__main__":
    main()
