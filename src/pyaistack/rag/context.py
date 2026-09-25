"""Bounded context construction and source presentation."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace

from .types import Citation, SearchResult

_SEPARATOR = "\n\n---\n\n"


def build_context(
    sources: Sequence[SearchResult],
    limit: int,
) -> tuple[str, tuple[SearchResult, ...]]:
    """Return rendered context and exact text excerpts actually included."""
    blocks: list[str] = []
    included: list[SearchResult] = []
    used = 0
    for result in sources:
        metadata = ", ".join(
            f"{key}={value}" for key, value in sorted(result.document.metadata.items())
        )
        header = "[Context]" + (f" ({metadata})" if metadata else "") + "\n"
        separator = _SEPARATOR if blocks else ""
        remaining = limit - used - len(separator) - len(header)
        if remaining <= 0:
            continue
        excerpt = result.document.text[:remaining]
        if not excerpt.strip():
            continue
        block = header + excerpt
        blocks.append(block)
        used += len(separator) + len(block)
        included.append(
            replace(
                result,
                document=replace(
                    result.document, text=excerpt, metadata=deepcopy(result.document.metadata)
                ),
            )
        )
    return _SEPARATOR.join(blocks), tuple(included)


def append_sources(answer: str, sources: Sequence[SearchResult]) -> str:
    """Display context provenance; this does not verify generated claims."""
    lines = ["Sources:"]
    for result in sources:
        metadata = result.document.metadata
        source = str(metadata.get("source", result.document.id))
        details = []
        if "chunk_index" in metadata:
            details.append(f"chunk {metadata['chunk_index']}")
        details.append(f"score {result.score:.4f}")
        if result.ranking_score is not None:
            details.append(f"fusion {result.ranking_score:.6f}")
        if result.rerank_score is not None:
            details.append(f"rerank {result.rerank_score:.6f}")
        lines.append(f"- {source} ({', '.join(details)})")
    return f"{answer.rstrip()}\n\n" + "\n".join(lines)


def build_citations(sources: Sequence[SearchResult]) -> tuple[Citation, ...]:
    """Create deterministic citations for excerpts included in model context."""
    citations: list[Citation] = []
    for result in sources:
        metadata = result.document.metadata
        chunk_index = metadata.get("chunk_index")
        citations.append(
            Citation(
                document_id=result.document.id,
                source=str(metadata.get("source", result.document.id)),
                chunk_index=chunk_index if type(chunk_index) is int else None,
                score=result.score,
                ranking_score=result.ranking_score,
                rerank_score=result.rerank_score,
            )
        )
    return tuple(citations)


def append_citations(answer: str, citations: Sequence[Citation]) -> str:
    """Append only citations verified from the context passed to the model."""
    lines = ["Citations:"]
    for citation in citations:
        details = [f"score {citation.score:.4f}"]
        if citation.chunk_index is not None:
            details.insert(0, f"chunk {citation.chunk_index}")
        if citation.ranking_score is not None:
            details.append(f"fusion {citation.ranking_score:.6f}")
        if citation.rerank_score is not None:
            details.append(f"rerank {citation.rerank_score:.6f}")
        lines.append(f"- {citation.source} ({', '.join(details)})")
    return f"{answer.rstrip()}\n\n" + "\n".join(lines)
