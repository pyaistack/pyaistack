"""Small, dependency-free retrieval evaluation utilities."""

from .retrieval import (
    RetrievalCase,
    RetrievalMetrics,
    RetrievalOutcome,
    RetrievalReport,
    evaluate_retrieval,
    evaluate_retrieval_report,
)

__all__ = [
    "RetrievalCase",
    "RetrievalMetrics",
    "RetrievalOutcome",
    "RetrievalReport",
    "evaluate_retrieval",
    "evaluate_retrieval_report",
]
