"""Framework-specific exceptions and user-facing error formatting."""

from __future__ import annotations

from pathlib import Path
from traceback import extract_tb


class AIStackError(Exception):
    """Base exception for this package."""


class ConfigurationError(AIStackError):
    """Raised when configuration is invalid."""


class DependencyError(AIStackError):
    """Raised when a required runtime dependency is unavailable."""


class ProviderError(AIStackError):
    """Raised when an embedding or chat provider fails."""


class RerankerError(AIStackError):
    """Raised when a reranker returns invalid scores or cannot complete."""


class VectorStoreError(AIStackError):
    """Raised for vector-store validation or execution failures."""


class VectorStoreConnectionError(VectorStoreError):
    """Raised when a vector-store database connection fails."""


class VectorStoreConstraintError(VectorStoreError):
    """Raised when a vector-store database constraint is violated."""


class VectorStoreSchemaError(VectorStoreError):
    """Raised when a vector-store database schema is incompatible."""


class VectorStoreTimeoutError(VectorStoreError):
    """Raised when a vector-store database operation times out."""


class EmptyIndexError(AIStackError):
    """Raised when retrieval is requested before documents are indexed."""


class LoaderError(AIStackError):
    """Raised when a document cannot be loaded."""


class MetadataFactoryError(LoaderError):
    """Raised when document metadata cannot be generated or validated."""


class UnsupportedFileTypeError(LoaderError):
    """Raised when a requested loader file type is not implemented."""


class ChunkingError(AIStackError):
    """Raised when chunking configuration or input is invalid."""


def format_error(error: AIStackError) -> str:
    """Render an AIStack error using the first application-code traceback frame.

    Applications can print this at their entry point to show a concise,
    standard Python-style traceback without PyAIStack implementation frames.
    """

    package_directory = Path(__file__).resolve().parent
    frame = next(
        (
            candidate
            for candidate in extract_tb(error.__traceback__)
            if package_directory not in Path(candidate.filename).resolve().parents
        ),
        None,
    )

    lines = ["Traceback (most recent call last):"]
    if frame is not None:
        lines.append(f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}')
        if frame.line:
            lines.append(f"    {frame.line.strip()}")
    lines.append(f"{type(error).__module__}.{type(error).__name__}: {error}")
    return "\n".join(lines)
