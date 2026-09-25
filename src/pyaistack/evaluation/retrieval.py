"""Deterministic retrieval measurements for a labeled evaluation set."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from ..rag.runtime import RAG


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    """One query and the document identifiers expected to answer it."""

    query: str
    relevant_document_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query cannot be empty")
        if not self.relevant_document_ids or any(
            not isinstance(document_id, str) or not document_id
            for document_id in self.relevant_document_ids
        ):
            raise ValueError("relevant_document_ids must contain document identifiers")


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Aggregate retrieval metrics for a labeled set of queries."""

    case_count: int
    recall_at_k: float
    mean_reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    """Per-query retrieval result for diagnosing a labeled evaluation run."""

    query: str
    relevant_document_ids: tuple[str, ...]
    returned_document_ids: tuple[str, ...]
    first_relevant_rank: int | None
    duration_ms: float


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    """JSON-compatible aggregate metrics and per-query retrieval outcomes."""

    metrics: RetrievalMetrics
    outcomes: tuple[RetrievalOutcome, ...]
    top_k: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report suitable for application-owned storage."""
        return {
            "metrics": {
                "case_count": self.metrics.case_count,
                "recall_at_k": self.metrics.recall_at_k,
                "mean_reciprocal_rank": self.metrics.mean_reciprocal_rank,
            },
            "top_k": self.top_k,
            "outcomes": [
                {
                    "query": outcome.query,
                    "relevant_document_ids": list(outcome.relevant_document_ids),
                    "returned_document_ids": list(outcome.returned_document_ids),
                    "first_relevant_rank": outcome.first_relevant_rank,
                    "duration_ms": outcome.duration_ms,
                }
                for outcome in self.outcomes
            ],
        }


def evaluate_retrieval(
    rag: RAG,
    cases: Iterable[RetrievalCase],
    *,
    top_k: int | None = None,
) -> RetrievalMetrics:
    """Measure recall@k and MRR using IDs returned by ``RAG.search``."""
    return evaluate_retrieval_report(rag, cases, top_k=top_k).metrics


def evaluate_retrieval_report(
    rag: RAG,
    cases: Iterable[RetrievalCase],
    *,
    top_k: int | None = None,
) -> RetrievalReport:
    """Return aggregate metrics and per-query outcomes for a labeled set."""
    values = tuple(cases)
    if not values:
        raise ValueError("cases cannot be empty")

    recalled = 0
    reciprocal_rank_total = 0.0
    outcomes: list[RetrievalOutcome] = []
    for case in values:
        started = perf_counter()
        results = rag.search(case.query, top_k=top_k)
        matches = [
            position
            for position, result in enumerate(results, start=1)
            if result.document.id in case.relevant_document_ids
        ]
        if matches:
            recalled += 1
            reciprocal_rank_total += 1.0 / matches[0]
        outcomes.append(
            RetrievalOutcome(
                query=case.query,
                relevant_document_ids=tuple(sorted(case.relevant_document_ids)),
                returned_document_ids=tuple(result.document.id for result in results),
                first_relevant_rank=matches[0] if matches else None,
                duration_ms=(perf_counter() - started) * 1_000,
            )
        )

    total = len(values)
    return RetrievalReport(
        metrics=RetrievalMetrics(
            case_count=total,
            recall_at_k=recalled / total,
            mean_reciprocal_rank=reciprocal_rank_total / total,
        ),
        outcomes=tuple(outcomes),
        top_k=top_k,
    )
