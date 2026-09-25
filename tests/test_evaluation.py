from __future__ import annotations

import pytest

from pyaistack import RAG, RAGConfig
from pyaistack.evaluation import RetrievalCase, evaluate_retrieval, evaluate_retrieval_report
from pyaistack.vectorstores import InMemoryVectorStore


class Embeddings:
    model_name = "test"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "lambda" in text.lower() else [0.0, 1.0] for text in texts]


class Chat:
    model_name = "test"

    def chat(self, messages):
        return "answer"


def test_evaluate_retrieval_reports_recall_and_mrr() -> None:
    rag = RAG(
        embedding_provider=Embeddings(),
        chat_provider=Chat(),
        vector_store=InMemoryVectorStore(),
        config=RAGConfig(top_k=2),
    )
    first, second = rag.add(["Lambda function", "S3 bucket"])

    metrics = evaluate_retrieval(
        rag,
        [
            RetrievalCase("lambda", frozenset({first.id})),
            RetrievalCase("s3", frozenset({second.id})),
        ],
    )

    assert metrics.case_count == 2
    assert metrics.recall_at_k == 1.0
    assert metrics.mean_reciprocal_rank == 1.0


def test_retrieval_case_and_empty_evaluation_are_validated() -> None:
    with pytest.raises(ValueError, match="query"):
        RetrievalCase("", frozenset({"a"}))
    with pytest.raises(ValueError, match="cases"):
        evaluate_retrieval(None, [])  # type: ignore[arg-type]


def test_retrieval_report_contains_per_query_outcomes() -> None:
    rag = RAG(
        embedding_provider=Embeddings(),
        chat_provider=Chat(),
        vector_store=InMemoryVectorStore(),
    )
    document = rag.add("Lambda function")[0]

    report = evaluate_retrieval_report(
        rag,
        [RetrievalCase("lambda", frozenset({document.id}))],
        top_k=1,
    )

    assert report.metrics.recall_at_k == 1.0
    assert report.outcomes[0].first_relevant_rank == 1
    assert report.to_dict()["outcomes"][0]["returned_document_ids"] == [document.id]
