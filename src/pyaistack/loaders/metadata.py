"""Contracts and validation shared by document metadata factories."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from ..exceptions import MetadataFactoryError
from .types import LoadedDocument

_RESERVED_METADATA_KEYS = frozenset({"source", "chunk_index", "chunk_start", "chunk_end"})


class MetadataFactory(Protocol):
    """Generate JSON-compatible metadata for one loaded source file."""

    def create(self, path: Path, text: str) -> Mapping[str, Any]: ...


def merge_generated_metadata(
    document: LoadedDocument, generated_metadata: Mapping[str, Any]
) -> LoadedDocument:
    """Validate and merge generated metadata without replacing loader fields."""

    if not isinstance(generated_metadata, Mapping):
        raise MetadataFactoryError("metadata factory must return a mapping")

    additional = dict(generated_metadata)
    reserved = _RESERVED_METADATA_KEYS.intersection(additional)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise MetadataFactoryError(f"metadata factory cannot set reserved metadata: {names}")
    if any(not isinstance(key, str) for key in additional):
        raise MetadataFactoryError("metadata factory keys must be strings")
    try:
        json.dumps(additional, allow_nan=False)
    except (TypeError, ValueError):
        raise MetadataFactoryError("metadata factory values must be JSON-compatible") from None

    return LoadedDocument(
        text=document.text, metadata=deepcopy({**document.metadata, **additional})
    )
