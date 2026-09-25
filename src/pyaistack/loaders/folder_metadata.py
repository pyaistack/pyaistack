"""Metadata generation derived from a document's directory structure."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..exceptions import MetadataFactoryError


class FolderMetadataFactory:
    """Create deterministic metadata from folders relative to a configured root.

    Folder names are mapped in order to ``folder_fields``. The file stem becomes
    ``title`` unless ``include_title`` is disabled.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        folder_fields: Sequence[str] = ("category", "topic"),
        include_title: bool = True,
    ) -> None:
        if any(not field.strip() for field in folder_fields):
            raise MetadataFactoryError("folder_fields must contain non-empty names")
        self.root = Path(root).resolve()
        self.folder_fields = tuple(folder_fields)
        self.include_title = include_title

    def create(self, path: Path, text: str) -> Mapping[str, Any]:
        """Create metadata from the source path; text is not needed for this factory."""

        del text
        try:
            relative_path = path.resolve().relative_to(self.root)
        except ValueError:
            raise MetadataFactoryError(f"source file is outside metadata root: {path}") from None

        metadata = {
            field: folder
            for field, folder in zip(self.folder_fields, relative_path.parts[:-1], strict=False)
        }
        if self.include_title:
            metadata["title"] = _title_from_stem(relative_path.stem)
        return metadata


def _title_from_stem(stem: str) -> str:
    return re.sub(r"[_-]+", " ", stem).strip()
