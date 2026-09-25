from __future__ import annotations

import pytest

from pyaistack.chunking import TextChunker
from pyaistack.exceptions import ChunkingError, MetadataFactoryError, UnsupportedFileTypeError
from pyaistack.loaders import (
    CSVMetadataFactory,
    DirectoryLoader,
    FolderMetadataFactory,
    LLMMetadataFactory,
    LoadedDocument,
    TextLoader,
)


def test_text_and_directory_loaders_preserve_source_metadata(tmp_path) -> None:
    first = tmp_path / "first.txt"
    first.write_text("First document.", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    second = nested / "second.txt"
    second.write_text("Second document.", encoding="utf-8")

    assert TextLoader(first).load()[0].metadata == {"source": str(first)}
    documents = DirectoryLoader(tmp_path, file_type="text").load()
    assert [document.text for document in documents] == ["First document.", "Second document."]


def test_directory_loader_requires_a_supported_file_type(tmp_path) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        DirectoryLoader(tmp_path, file_type="pdf")


def test_directory_loader_generates_folder_metadata_one_file_at_a_time(tmp_path) -> None:
    finance = tmp_path / "finance" / "budgeting"
    finance.mkdir(parents=True)
    document_path = finance / "monthly-budget.txt"
    document_path.write_text("Budget guidance.", encoding="utf-8")

    loader = DirectoryLoader(
        tmp_path,
        file_type="text",
        metadata_factory=FolderMetadataFactory(tmp_path),
    )

    documents = list(loader.iter_load())

    assert documents[0].metadata == {
        "source": str(document_path),
        "category": "finance",
        "topic": "budgeting",
        "title": "monthly budget",
    }


def test_directory_loader_reads_metadata_from_a_csv_manifest(tmp_path) -> None:
    finance = tmp_path / "finance"
    finance.mkdir()
    document_path = finance / "budget.txt"
    document_path.write_text("Budget guidance.", encoding="utf-8")
    manifest = tmp_path / "metadata.csv"
    manifest.write_text(
        'source,metadata_json\n'
        'finance/budget.txt,"{""category"": [""finance""], ""tags"": [""budget""]}"\n',
        encoding="utf-8",
    )

    documents = DirectoryLoader(
        tmp_path,
        file_type="text",
        metadata_factory=CSVMetadataFactory(manifest, root=tmp_path),
    ).load()

    assert documents[0].metadata == {
        "source": str(document_path),
        "category": ["finance"],
        "tags": ["budget"],
    }


def test_csv_metadata_factory_requires_a_manifest_entry_for_every_source(tmp_path) -> None:
    (tmp_path / "guide.txt").write_text("Guide.", encoding="utf-8")
    manifest = tmp_path / "metadata.csv"
    manifest.write_text("source,metadata_json\n", encoding="utf-8")
    loader = DirectoryLoader(
        tmp_path,
        file_type="text",
        metadata_factory=CSVMetadataFactory(manifest, root=tmp_path),
    )

    with pytest.raises(MetadataFactoryError, match="no CSV metadata"):
        loader.load()


class FakeMetadataChatProvider:
    model_name = "fake-chat"

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[dict[str, str]] | None = None

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return self.response


def test_llm_metadata_factory_uses_a_bounded_content_sample(tmp_path) -> None:
    chat_provider = FakeMetadataChatProvider(
        '{"category": ["Finance Planning"], "audience": "Businesses, General Public", '
        '"language": "English", "tags": ["Budget", "monthly review"]}'
    )
    factory = LLMMetadataFactory(
        chat_provider,
        fields=("category", "audience", "language", "tags"),
        max_sentences=2,
    )

    metadata = factory.create(
        tmp_path / "budget.txt", "First sentence. Second sentence. Third sentence."
    )

    assert metadata == {
        "category": ["finance_planning"],
        "audience": ["businesses", "general_public"],
        "language": ["en"],
        "tags": ["budget", "monthly_review"],
    }
    assert chat_provider.messages is not None
    assert "First sentence. Second sentence." in chat_provider.messages[1]["content"]
    assert "Third sentence." not in chat_provider.messages[1]["content"]


def test_llm_metadata_factory_rejects_invalid_json(tmp_path) -> None:
    factory = LLMMetadataFactory(FakeMetadataChatProvider("not JSON"))

    with pytest.raises(MetadataFactoryError, match="invalid JSON"):
        factory.create(tmp_path / "guide.txt", "Document content.")


def test_text_chunker_adds_deterministic_chunk_metadata() -> None:
    chunks = TextChunker(chunk_size=4, chunk_overlap=1).chunk(
        [LoadedDocument(text="abcdefgh", metadata={"source": "note.txt"})]
    )
    assert [chunk.text for chunk in chunks] == ["abcd", "defg", "gh"]
    assert chunks[1].metadata == {
        "source": "note.txt",
        "chunk_index": 2,
        "chunk_start": 3,
        "chunk_end": 7,
    }


def test_text_chunker_validates_overlap() -> None:
    with pytest.raises(ChunkingError):
        TextChunker(chunk_size=10, chunk_overlap=10)


def test_text_chunker_prefers_sentence_boundaries() -> None:
    chunks = TextChunker(chunk_size=26, chunk_overlap=0).chunk(
        [LoadedDocument(text="First sentence is here. Second sentence is here. Third sentence.")]
    )

    assert [chunk.text for chunk in chunks] == [
        "First sentence is here. ",
        "Second sentence is here. ",
        "Third sentence.",
    ]
    assert TextChunker().separators == ("\n\n", "\n", ". ")
