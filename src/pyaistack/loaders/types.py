"""Data types shared by document loaders and chunkers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FileType(str, Enum):  # noqa: UP042 - StrEnum is unavailable on Python 3.10.
    """File types supported by loaders in this release."""

    TEXT = "text"


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """Text and metadata produced by a loader before indexing."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
