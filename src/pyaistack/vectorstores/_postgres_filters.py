"""Safe PostgreSQL JSONB predicates matching PyAIStack metadata semantics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..exceptions import VectorStoreError


def compile_filter(
    metadata_column: str,
    metadata_filter: Mapping[str, Any] | None,
) -> tuple[str, list[Any]]:
    """Compile an already-quoted metadata column into SQL and bound values."""

    if metadata_filter is None or not metadata_filter:
        return "TRUE", []
    if not isinstance(metadata_filter, Mapping):
        raise VectorStoreError("metadata_filter must be a mapping")

    predicates: list[str] = []
    parameters: list[Any] = []
    for key, value in metadata_filter.items():
        if not isinstance(key, str):
            raise VectorStoreError("metadata filter keys must be strings")
        try:
            encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise VectorStoreError("metadata filter values must be JSON-compatible") from error

        if isinstance(value, list):
            if value:
                scalar_placeholders = ",".join("%s::jsonb" for _ in value)
                scalar_predicate = f"{metadata_column} -> %s IN ({scalar_placeholders})"
                scalar_key_parameters = [key]
                scalar_parameters = [
                    json.dumps(item, allow_nan=False, separators=(",", ":")) for item in value
                ]
            else:
                scalar_predicate = "FALSE"
                scalar_key_parameters = []
                scalar_parameters = []
            predicates.append(
                "("
                f"{metadata_column} ? %s AND ("
                f"(jsonb_typeof({metadata_column} -> %s) = 'array' "
                f"AND {metadata_column} -> %s @> %s::jsonb) OR "
                f"(jsonb_typeof({metadata_column} -> %s) <> 'array' "
                f"AND {scalar_predicate})"
                ")"
                ")"
            )
            parameters.extend(
                [key, key, key, encoded, key, *scalar_key_parameters, *scalar_parameters]
            )
        else:
            encoded_array = json.dumps([value], allow_nan=False, separators=(",", ":"))
            predicates.append(
                "("
                f"{metadata_column} ? %s AND ("
                f"(jsonb_typeof({metadata_column} -> %s) = 'array' "
                f"AND {metadata_column} -> %s @> %s::jsonb) OR "
                f"(jsonb_typeof({metadata_column} -> %s) <> 'array' "
                f"AND {metadata_column} -> %s = %s::jsonb)"
                ")"
                ")"
            )
            parameters.extend([key, key, key, encoded_array, key, key, encoded])
    return " AND ".join(predicates), parameters
