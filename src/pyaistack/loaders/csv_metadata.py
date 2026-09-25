"""Metadata generation from a CSV manifest keyed by source path."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..exceptions import MetadataFactoryError
from .metadata import merge_generated_metadata
from .types import LoadedDocument


class CSVMetadataFactory:
    """Read JSON metadata for root-relative source paths from a CSV manifest.

    The manifest must contain ``source`` and ``metadata_json`` columns. Each
    source path is relative to ``root`` so the manifest remains portable.
    """

    def __init__(self, csv_path: str | Path, *, root: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.root = Path(root).resolve()
        self._metadata_by_source = self._read_manifest()

    def create(self, path: Path, text: str) -> Mapping[str, Any]:
        """Return manifest metadata for one source file; text is not needed."""

        del text
        source = self._relative_source(path)
        try:
            return deepcopy(self._metadata_by_source[source])
        except KeyError:
            raise MetadataFactoryError(f"no CSV metadata found for source: {source}") from None

    def _read_manifest(self) -> dict[str, dict[str, Any]]:
        try:
            with self.csv_path.open(encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file, strict=True)
                self._validate_headers(reader.fieldnames)
                metadata_by_source: dict[str, dict[str, Any]] = {}
                for line_number, row in enumerate(reader, start=2):
                    if None in row or any(value is None for value in row.values()):
                        raise MetadataFactoryError(f"invalid CSV row shape at line {line_number}")
                    source = self._source_from_row(row, line_number)
                    if source in metadata_by_source:
                        raise MetadataFactoryError(
                            f"duplicate CSV metadata source at line {line_number}: {source}"
                        )
                    metadata_by_source[source] = self._metadata_from_row(row, line_number)
                return metadata_by_source
        except (OSError, UnicodeError):
            raise MetadataFactoryError(f"could not read metadata CSV: {self.csv_path}") from None
        except csv.Error:
            raise MetadataFactoryError(f"could not parse metadata CSV: {self.csv_path}") from None

    @staticmethod
    def _validate_headers(headers: list[str] | None) -> None:
        required = {"source", "metadata_json"}
        if headers is None or set(headers) != required or len(headers) != 2:
            raise MetadataFactoryError("metadata CSV must contain source and metadata_json columns")

    def _source_from_row(self, row: dict[str, str | None], line_number: int) -> str:
        source = row.get("source")
        if source is None or not source.strip():
            raise MetadataFactoryError(f"metadata CSV source is empty at line {line_number}")
        if (
            Path(source).is_absolute()
            or ":" in source
            or ".." in source.replace("\\", "/").split("/")
        ):
            raise MetadataFactoryError(f"CSV source must be root-relative at line {line_number}")
        try:
            return self._relative_source(self.root / source.replace("\\", "/"))
        except MetadataFactoryError:
            raise MetadataFactoryError(
                f"metadata CSV source must be inside root at line {line_number}: {source}"
            ) from None

    @staticmethod
    def _metadata_from_row(row: dict[str, str | None], line_number: int) -> dict[str, Any]:
        value = row.get("metadata_json")
        if value is None or not value.strip():
            raise MetadataFactoryError(f"metadata_json is empty at line {line_number}")
        try:
            metadata = json.loads(value, object_pairs_hook=_unique_object)
        except ValueError:
            raise MetadataFactoryError(f"metadata_json is invalid at line {line_number}") from None
        if not isinstance(metadata, dict):
            raise MetadataFactoryError(f"metadata_json must be an object at line {line_number}")
        return merge_generated_metadata(LoadedDocument(text=""), metadata).metadata

    def _relative_source(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            raise MetadataFactoryError(f"source file is outside metadata root: {path}") from None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate metadata JSON key")
        result[key] = value
    return result
