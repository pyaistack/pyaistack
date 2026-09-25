"""Separator-aware character-based text chunking."""

from __future__ import annotations

from collections.abc import Sequence

from ..exceptions import ChunkingError
from ..loaders.types import LoadedDocument

_DEFAULT_SEPARATORS = ("\n\n", "\n", ". ")


class TextChunker:
    """Split text at paragraph, line, and sentence boundaries when possible."""

    def __init__(
        self, *, chunk_size: int = 1_000, chunk_overlap: int = 150,
        separators: Sequence[str] = _DEFAULT_SEPARATORS,
    ) -> None:
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ChunkingError("chunk_size must be greater than zero")
        if type(chunk_overlap) is not int or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be non-negative and smaller than chunk_size")
        if isinstance(separators, str) or not separators or any(
            not isinstance(separator, str) or not separator for separator in separators
        ):
            raise ChunkingError("separators must contain only non-empty values")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = tuple(separators)

    def chunk(self, documents: Sequence[LoadedDocument]) -> tuple[LoadedDocument, ...]:
        chunks: list[LoadedDocument] = []
        for document in documents:
            text = document.text.strip()
            if not text:
                raise ChunkingError("documents cannot contain empty text")
            for index, (start, end) in enumerate(self._chunk_spans(text, 0), start=1):
                metadata = dict(document.metadata)
                metadata.update({"chunk_index": index, "chunk_start": start, "chunk_end": end})
                chunks.append(LoadedDocument(text[start:end], metadata))
        return tuple(chunks)

    def _chunk_spans(self, text: str, separator_index: int) -> list[tuple[int, int]]:
        if len(text) <= self.chunk_size:
            return [(0, len(text))]
        for index in range(separator_index, len(self.separators)):
            separator = self.separators[index]
            if separator not in text:
                continue
            spans: list[tuple[int, int]] = []
            offset = 0
            for part in self._parts(text, separator):
                if len(part) <= self.chunk_size:
                    spans.append((offset, offset + len(part)))
                else:
                    spans.extend(
                        (offset + start, offset + end)
                        for start, end in self._chunk_spans(part, index + 1)
                    )
                offset += len(part)
            return self._merge(spans)
        return self._hard_cuts(len(text))

    @staticmethod
    def _parts(text: str, separator: str) -> list[str]:
        parts: list[str] = []
        start = 0
        while (position := text.find(separator, start)) != -1:
            end = position + len(separator)
            parts.append(text[start:end])
            start = end
        parts.append(text[start:])
        return parts

    def _merge(self, spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        limit = self.chunk_size - self.chunk_overlap
        start: int | None = None
        end: int | None = None
        for part_start, part_end in spans:
            if start is None:
                start, end = part_start, part_end
            elif part_end - start <= limit:
                end = part_end
            else:
                merged.append((start, end))
                start, end = part_start, part_end
        if start is not None and end is not None:
            merged.append((start, end))
        result: list[tuple[int, int]] = []
        for index, (start, end) in enumerate(merged):
            if index:
                start = max(start, merged[index - 1][1] - self.chunk_overlap, end - self.chunk_size)
            result.append((start, end))
        return result

    def _hard_cuts(self, text_length: int) -> list[tuple[int, int]]:
        step = self.chunk_size - self.chunk_overlap
        return [
            (start, min(start + self.chunk_size, text_length))
            for start in range(0, text_length, step)
        ]
