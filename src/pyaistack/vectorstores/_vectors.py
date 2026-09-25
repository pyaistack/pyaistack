"""Shared vector validation, normalization, and cosine arithmetic."""

import math
from collections.abc import Sequence

from ..exceptions import VectorStoreError


def normalize(vector: Sequence[float]) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in vector)
    except (ValueError, TypeError, OverflowError) as error:
        raise VectorStoreError("embedding vector must contain numbers") from error
    if not values or any(not math.isfinite(value) for value in values):
        raise VectorStoreError("embedding vector must contain finite values and cannot be empty")
    scale = max(abs(value) for value in values)
    if scale == 0:
        raise VectorStoreError("embedding vector cannot have zero magnitude")
    scaled = tuple(value / scale for value in values)
    magnitude = math.sqrt(math.fsum(value * value for value in scaled))
    return tuple(value / magnitude for value in scaled)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise VectorStoreError("query dimension does not match index dimension")
    if any(not math.isfinite(value) for value in right):
        raise VectorStoreError("stored vector contains non-finite values")
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(left, right, strict=True))))


def validate_search(top_k: int, min_score: float | None) -> None:
    if type(top_k) is not int or top_k <= 0:
        raise VectorStoreError("top_k must be a positive integer")
    if min_score is not None and (
        type(min_score) not in (int, float)
        or not math.isfinite(min_score)
        or not -1 <= min_score <= 1
    ):
        raise VectorStoreError("min_score must be finite and between -1 and 1")
