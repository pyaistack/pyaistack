from __future__ import annotations

from collections.abc import Sequence

import pytest

from pyaistack import RAG, Document, RAGConfig, SearchResult
from pyaistack.chunking import TextChunker
from pyaistack.exceptions import EmptyIndexError, RerankerError, VectorStoreError
from pyaistack.loaders import LoadedDocument
from pyaistack.rag.types import Message
from pyaistack.vectorstores import InMemoryVectorStore


class FakeEmbeddingProvider:
    model_name = "fake-embedding"

    _vectors = {
        "lambda": [1.0, 0.0],
        "s3": [0.0, 1.0],
        "serverless": [0.9, 0.1],
    }

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        result = []
        for text in texts:
            lowered = text.lower()
            if "lambda" in lowered or "serverless" in lowered:
                vector_name = "lambda" if "lambda" in lowered else "serverless"
                result.append(self._vectors[vector_name])
            else:
                result.append(self._vectors["s3"])
        return result


class FakeChatProvider:
    model_name = "fake-chat"

    def __init__(self) -> None:
        self.last_messages: Sequence[Message] | None = None

    def chat(self, messages: Sequence[Message]) -> str:
        self.last_messages = messages
        return "AWS Lambda."


class ReverseReranker:
    def rerank(self, query: str, candidates: Sequence[SearchResult]) -> Sequence[float]:
        return list(range(len(candidates)))


def build_rag() -> tuple[RAG, FakeChatProvider]:
    chat = FakeChatProvider()
    rag = RAG(
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=chat,
        vector_store=InMemoryVectorStore(),
        config=RAGConfig(top_k=1),
    )
    return rag, chat


def test_add_search_and_ask() -> None:
    rag, chat = build_rag()
    rag.add(
        ["AWS Lambda runs serverless code.", "Amazon S3 stores objects."],
        metadatas=[{"source": "lambda"}, {"source": "s3"}],
    )

    results = rag.search("serverless", top_k=1)
    assert len(results) == 1
    assert "Lambda" in results[0].document.text

    answer = rag.ask("What runs serverless code?")
    assert answer.text.startswith("AWS Lambda.")
    assert answer.sources[0].document.metadata["source"] == "lambda"
    assert chat.last_messages is not None
    assert "[Context]" in chat.last_messages[-1]["content"]
    assert "Sources:" not in answer.text


def test_empty_index_is_explicit() -> None:
    rag, _ = build_rag()
    with pytest.raises(EmptyIndexError):
        rag.search("serverless")


def test_metadata_length_is_validated() -> None:
    rag, _ = build_rag()
    with pytest.raises(ValueError, match="metadatas"):
        rag.add(["one", "two"], metadatas=[{"x": 1}])


def test_metadata_filter_limits_search_and_ask_results() -> None:
    rag, chat = build_rag()
    rag.add(
        ["AWS Lambda runs serverless code.", "Amazon S3 stores objects."],
        metadatas=[
            {"source": "lambda", "category": ["compute", "cloud"]},
            {"source": "s3", "category": ["storage"]},
        ],
    )

    results = rag.search("serverless", metadata_filter={"category": "storage"})
    assert [result.document.metadata["source"] for result in results] == ["s3"]

    answer = rag.ask("What is available?", metadata_filter={"category": "storage"})
    assert [result.document.metadata["source"] for result in answer.sources] == ["s3"]
    assert chat.last_messages is not None
    assert "Amazon S3" in chat.last_messages[-1]["content"]

    results = rag.search("serverless", metadata_filter={"category": ["cloud", "compute"]})
    assert [result.document.metadata["source"] for result in results] == ["lambda"]


def test_metadata_filter_must_be_a_mapping() -> None:
    rag, _ = build_rag()
    rag.add("AWS Lambda runs serverless code.")

    with pytest.raises(ValueError, match="metadata_filter"):
        rag.search("serverless", metadata_filter=[("category", "compute")])  # type: ignore[arg-type]


def test_vector_store_rejects_dimension_change() -> None:
    store = InMemoryVectorStore()
    rag, _ = build_rag()
    document = rag.add("AWS Lambda runs serverless code.")[0]
    store.add([document], [[1.0, 0.0]])

    with pytest.raises(VectorStoreError, match="dimension changed"):
        store.add([document], [[1.0, 0.0, 0.0]])


def test_clear_resets_index() -> None:
    rag, _ = build_rag()
    rag.add("AWS Lambda runs serverless code.")
    assert rag.document_count == 1
    rag.clear()
    assert rag.document_count == 0


def test_add_documents_applies_the_configured_chunker() -> None:
    chat = FakeChatProvider()
    rag = RAG(
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=chat,
        vector_store=InMemoryVectorStore(),
        chunker=TextChunker(chunk_size=10, chunk_overlap=0),
    )

    documents = rag.add_documents([LoadedDocument(text="AWS Lambda runs serverless code.")])

    assert len(documents) == 4
    assert documents[0].metadata["chunk_index"] == 1


def test_context_includes_chunk_metadata_without_a_model_citation_label() -> None:
    rag, _ = build_rag()
    context = rag._build_context(
        [
            SearchResult(
                document=Document(id="one", text="Indexed content.", metadata={"chunk_index": 5}),
                score=0.9,
            )
        ]
    )

    assert context.startswith("[Context] (chunk_index=5)")


def test_include_sources_appends_verified_sources_to_answer_text() -> None:
    chat = FakeChatProvider()
    rag = RAG(
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=chat,
        vector_store=InMemoryVectorStore(),
        config=RAGConfig(include_sources=True),
    )
    rag.add("AWS Lambda runs serverless code.", metadatas=[{"source": "lambda"}])

    answer = rag.ask("What runs serverless code?")

    assert "Sources:\n- lambda (score" in answer.text


def test_reranker_scores_candidates_and_preserves_original_similarity() -> None:
    rag = RAG(
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=FakeChatProvider(),
        vector_store=InMemoryVectorStore(),
        config=RAGConfig(top_k=1, rerank_candidate_k=2),
        reranker=ReverseReranker(),
    )
    rag.add(
        ["AWS Lambda runs serverless code.", "Amazon S3 stores objects."],
        metadatas=[{"source": "lambda"}, {"source": "s3"}],
    )

    result = rag.search("serverless")

    assert result[0].document.metadata["source"] == "s3"
    assert result[0].score < 0.5
    assert result[0].rerank_score == 1.0


def test_invalid_reranker_scores_are_explicit() -> None:
    class BrokenReranker:
        def rerank(self, query: str, candidates: Sequence[SearchResult]) -> Sequence[float]:
            return [float("nan")]

    rag = RAG(
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=FakeChatProvider(),
        reranker=BrokenReranker(),
    )
    rag.add("AWS Lambda runs serverless code.")

    with pytest.raises(RerankerError, match="finite"):
        rag.search("serverless")


def test_citations_are_exact_context_provenance() -> None:
    rag = RAG(
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=FakeChatProvider(),
        config=RAGConfig(include_citations=True),
    )
    rag.add(
        "AWS Lambda runs serverless code.",
        metadatas=[{"source": "lambda.txt", "chunk_index": 3}],
    )

    answer = rag.ask("What runs serverless code?")

    assert answer.citations[0].source == "lambda.txt"
    assert answer.citations[0].chunk_index == 3
    assert answer.citations[0].document_id == answer.context_sources[0].document.id
    assert "Citations:\n- lambda.txt (chunk 3, score" in answer.text


def test_relevance_threshold_and_response_skip_chat_generation() -> None:
    chat = FakeChatProvider()
    rag = RAG(
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=chat,
        config=RAGConfig(
            min_score=0.5,
            insufficient_context_response="No relevant indexed context was found.",
        ),
    )
    rag.add("Amazon S3 stores objects.")

    answer = rag.ask("What runs serverless code?")

    assert answer.text == "No relevant indexed context was found."
    assert answer.sources == ()
    assert chat.last_messages is None


def test_system_prompt_suffix_preserves_the_grounding_prompt() -> None:
    chat = FakeChatProvider()
    rag = RAG(
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=chat,
        vector_store=InMemoryVectorStore(),
        system_prompt_suffix="Answer with concise bullet points.",
    )
    rag.add("AWS Lambda runs serverless code.")

    rag.ask("What runs serverless code?")

    assert chat.last_messages is not None
    prompt = chat.last_messages[0]["content"]
    assert "Answer the user's question using only the supplied context." in prompt
    assert prompt.endswith("Answer with concise bullet points.")
