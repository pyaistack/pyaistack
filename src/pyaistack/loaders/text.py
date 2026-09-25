"""UTF-8 plain-text document loader."""

from __future__ import annotations

from pathlib import Path

from ..exceptions import LoaderError
from .types import LoadedDocument


class TextLoader:
    """Load one UTF-8 `.txt` file with its path stored as metadata."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> tuple[LoadedDocument, ...]:
        if self.path.suffix.lower() != ".txt":
            raise LoaderError(f"text loader only supports .txt files: {self.path}")
        try:
            text = self.path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            raise LoaderError(f"could not read text file: {self.path}") from None
        if not text:
            raise LoaderError(f"text file is empty: {self.path}")
        return (LoadedDocument(text=text, metadata={"source": str(self.path)}),)
