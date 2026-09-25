"""Run a local RAG query with reranking, verified citations, and retrieval evaluation.

Start Ollama and pull the default models first:
    ollama pull embeddinggemma
    ollama pull gemma3:4b

Then run:
    python examples/reranking/in_memory_evaluation.py
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pyaistack import RAG, RAGConfig, SearchResult
from pyaistack.evaluation import RetrievalCase, evaluate_retrieval_report


class KeywordReranker:
    """Small transparent reranker for demonstration; replace in real applications."""

    def rerank(self, query: str, candidates: Sequence[SearchResult]) -> Sequence[float]:
        query_terms = set(re.findall(r"\w+", query.lower()))
        return [
            float(len(query_terms.intersection(re.findall(r"\w+", item.document.text.lower()))))
            for item in candidates
        ]


def main() -> None:
    rag = RAG(
        config=RAGConfig(
            top_k=2,
            rerank_candidate_k=4,
            min_score=0.15,
            min_context_results=1,
            include_citations=True,
            insufficient_context_response="No relevant policy information was found.",
        ),
        reranker=KeywordReranker(),
    )

    documents = rag.add(
        [
            "FIN-042 requires manager approval before a purchase above $500.",
            "FIN-043 covers travel expenses and requires receipts for reimbursement.",
            "HR-008 provides annual leave guidance for employees.",
            "SEC-018 explains the password reset process.",
        ],
        metadatas=[
            {"source": "finance-policies.txt", "chunk_index": 1, "category": "finance"},
            {"source": "finance-policies.txt", "chunk_index": 2, "category": "finance"},
            {"source": "hr-policies.txt", "chunk_index": 1, "category": "hr"},
            {"source": "security-policies.txt", "chunk_index": 1, "category": "security"},
        ],
    )

    answer = rag.ask("What approval is required for a purchase above $500?")
    print(answer.text)

    report = evaluate_retrieval_report(
        rag,
        [
            RetrievalCase("What approval is required for FIN-042?", frozenset({documents[0].id})),
            RetrievalCase("How do I reset my password?", frozenset({documents[3].id})),
        ],
        top_k=2,
    )
    print(
        f"\nEvaluation: recall@2={report.metrics.recall_at_k:.2f}, "
        f"MRR={report.metrics.mean_reciprocal_rank:.2f}"
    )
    for outcome in report.outcomes:
        print(f"First relevant rank: {outcome.first_relevant_rank}")


if __name__ == "__main__":
    main()
