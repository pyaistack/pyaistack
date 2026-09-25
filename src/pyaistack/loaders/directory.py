"""Directory loader for one explicitly selected file type."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..exceptions import LoaderError, MetadataFactoryError, UnsupportedFileTypeError
from .metadata import MetadataFactory, merge_generated_metadata
from .text import TextLoader
from .types import FileType, LoadedDocument


class DirectoryLoader:
    """Recursively load files of a required, explicitly selected type."""

    def __init__(
        self,
        path: str | Path,
        *,
        file_type: FileType | str,
        metadata_factory: MetadataFactory | None = None,
    ) -> None:
        self.path = Path(path)
        if metadata_factory is not None and not callable(
            getattr(metadata_factory, "create", None)
        ):
            raise MetadataFactoryError("metadata_factory must define create(path, text)")
        self.metadata_factory = metadata_factory
        try:
            self.file_type = FileType(file_type)
        except ValueError:
            raise UnsupportedFileTypeError(
                f"file type {file_type!r} is not supported; available: text"
            ) from None

    def load(self) -> tuple[LoadedDocument, ...]:
        """Load every matching file into memory in deterministic path order."""

        return tuple(self.iter_load())

    def iter_load(self) -> Iterator[LoadedDocument]:
        """Yield loaded documents one at a time in deterministic path order."""

        if not self.path.is_dir():
            raise LoaderError(f"directory does not exist: {self.path}")
        if self.file_type is not FileType.TEXT:  # Defensive for future enum members.
            raise UnsupportedFileTypeError(f"file type is not implemented: {self.file_type.value}")
        for path in sorted(self.path.rglob("*.txt")):
            document = TextLoader(path).load()[0]
            if self.metadata_factory is None:
                yield document
                continue
            try:
                generated_metadata = self.metadata_factory.create(path, document.text)
                yield merge_generated_metadata(document, generated_metadata)
            except MetadataFactoryError:
                raise
            except Exception as error:
                raise MetadataFactoryError(f"could not generate metadata for: {path}") from error
