"""Configuration objects for the RAG runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ..exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class RAGConfig:
    """Runtime retrieval and context settings."""

    top_k: int = 3
    min_score: float | None = None
    max_context_chars: int = 12_000
    include_sources: bool = False
    include_citations: bool = False
    insufficient_context_response: str = (
        "I don't have enough information in the indexed context to answer that."
    )
    min_context_results: int = 1
    retrieval_mode: Literal["vector", "hybrid"] = "vector"
    candidate_k: int = 20
    rerank_candidate_k: int = 20
    observability_metadata_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("top_k", "max_context_chars", "candidate_k", "rerank_candidate_k"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer")
        if self.retrieval_mode not in ("vector", "hybrid"):
            raise ConfigurationError("retrieval_mode must be vector or hybrid")
        if self.min_score is not None and (
            type(self.min_score) not in (int, float) or not math.isfinite(self.min_score)
        ):
            raise ConfigurationError("min_score must be a finite number or None")
        if self.min_score is not None and not -1.0 <= self.min_score <= 1.0:
            raise ConfigurationError("min_score must be between -1.0 and 1.0")
        for name in ("include_sources", "include_citations"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigurationError(f"{name} must be a boolean")
        if type(self.min_context_results) is not int or self.min_context_results <= 0:
            raise ConfigurationError("min_context_results must be a positive integer")
        if (
            not isinstance(self.insufficient_context_response, str)
            or not self.insufficient_context_response.strip()
        ):
            raise ConfigurationError("insufficient_context_response cannot be empty")
        if not isinstance(self.observability_metadata_keys, tuple) or any(
            not isinstance(key, str) or not key.strip() for key in self.observability_metadata_keys
        ):
            raise ConfigurationError(
                "observability_metadata_keys must be a tuple of non-empty strings"
            )
        if len(set(self.observability_metadata_keys)) != len(self.observability_metadata_keys):
            raise ConfigurationError("observability_metadata_keys cannot contain duplicates")
        if len(self.observability_metadata_keys) > 16:
            raise ConfigurationError("observability_metadata_keys cannot contain more than 16 keys")
