"""Compile scalar filters to SQL; preserve complex comparisons with the shared matcher."""
import json
from collections.abc import Mapping
from typing import Any

from ..exceptions import VectorStoreError


def compile_filter(metadata_filter: Mapping[str, Any] | None) -> tuple[str, list[Any]]:
    if metadata_filter is None:
        return "1", []
    if not isinstance(metadata_filter, Mapping):
        raise VectorStoreError("metadata_filter must be a mapping")
    try:
        encoded = json.dumps(dict(metadata_filter), allow_nan=False)
    except (ValueError, TypeError) as error:
        raise VectorStoreError("metadata_filter must be JSON-compatible") from error
    if any(not isinstance(key, str) for key in metadata_filter):
        raise VectorStoreError("metadata_filter keys must be strings")
    if any(isinstance(value, (list, dict, tuple)) for value in metadata_filter.values()):
        return "pyaistack_matches(metadata_json,?)", [encoded]
    clauses, parameters = [], []
    for key, value in metadata_filter.items():
        # json_each distinguishes absent keys from JSON null and avoids JSON path interpolation.
        clauses.append("""EXISTS (
            SELECT 1 FROM json_each(metadata_json) AS field WHERE field.key=? AND (
                (field.type NOT IN ('array','object') AND field.atom IS ?) OR
                (field.type='array' AND EXISTS (
                    SELECT 1 FROM json_each(field.value) AS item
                    WHERE item.type NOT IN ('array','object') AND item.atom IS ?
                ))
            )
        )""")
        parameters.extend((key, value, value))
    return " AND ".join(clauses) or "1", parameters
