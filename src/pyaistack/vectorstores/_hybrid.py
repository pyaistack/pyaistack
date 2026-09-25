"""Plain-text keyword queries and reciprocal rank fusion."""

import re
from collections.abc import Sequence
from dataclasses import replace

from ..rag.types import SearchResult


def keyword_query(query: str) -> str:
    # Bound lexical work; never interpret user text as FTS operators.
    terms = list(dict.fromkeys(re.findall(r"[^\W_]+", query, flags=re.UNICODE)))[:64]
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def fuse(
    vector: Sequence[SearchResult],
    keyword: Sequence[SearchResult],
    top_k: int,
    min_score: float | None,
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    documents: dict[str, SearchResult] = {}
    for branch in (vector, keyword):
        for rank, result in enumerate(branch, start=1):
            key = result.document.id
            documents[key] = result
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return [
        replace(documents[key], ranking_score=scores[key])
        for key in ordered
        if min_score is None or documents[key].score >= min_score
    ][:top_k]
