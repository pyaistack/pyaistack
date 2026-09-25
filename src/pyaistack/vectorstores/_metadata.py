"""Shared scalar and list metadata filtering for vector stores."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def metadata_matches(
    metadata: Mapping[str, Any], metadata_filter: Mapping[str, Any] | None
) -> bool:
    """Return whether metadata contains every requested matching key/value pair."""

    if metadata_filter is None:
        return True
    return all(
        key in metadata and _value_matches(metadata[key], value)
        for key, value in metadata_filter.items()
    )


def _value_matches(stored_value: Any, filter_value: Any) -> bool:
    if isinstance(stored_value, list):
        if isinstance(filter_value, list):
            return all(value in stored_value for value in filter_value)
        return filter_value in stored_value
    if isinstance(filter_value, list):
        return stored_value in filter_value
    return stored_value == filter_value
