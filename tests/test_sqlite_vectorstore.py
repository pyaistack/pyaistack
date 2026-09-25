from __future__ import annotations

import pytest

from pyaistack import RAG
from pyaistack.exceptions import VectorStoreError
from pyaistack.rag.types import Document
from pyaistack.vectorstores import SQLiteVectorStore


class FakeEmbeddingProvider:
    model_name = "embedding-a"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeChatProvider:
    model_name = "chat-a"

    def chat(self, messages: list[object]) -> str:
        return "answer"


def test_sqlite_store_persists_searchable_vectors_and_model_binding(tmp_path) -> None:
    path = tmp_path / "knowledge.db"
    document = Document(id="one", text="Lambda", metadata={"source": "lambda"})
    store = SQLiteVectorStore(path)
    store.set_embedding_model("embedding-a")
    store.add([document], [[1.0, 0.0]])
    store.close()

    reopened = SQLiteVectorStore(path)
    assert len(reopened) == 1
    assert reopened.search([1.0, 0.0], top_k=1)[0].document.metadata == {"source": "lambda"}
    with pytest.raises(VectorStoreError, match="embedding model changed"):
        reopened.set_embedding_model("embedding-b")
    reopened.clear()
    reopened.set_embedding_model("embedding-b")
    assert len(reopened) == 0


def test_rag_uses_an_empty_sqlite_store_instead_of_replacing_it(tmp_path) -> None:
    path = tmp_path / "knowledge.db"
    rag = RAG(
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=FakeChatProvider(),
        vector_store=SQLiteVectorStore(path),
    )

    rag.add("Persistent document.")

    reopened = SQLiteVectorStore(path)
    reopened.set_embedding_model("embedding-a")
    assert len(reopened) == 1


def test_sqlite_store_filters_candidates_before_scoring(tmp_path) -> None:
    store = SQLiteVectorStore(tmp_path / "knowledge.db")
    store.add(
        [
            Document(id="finance", text="Budget", metadata={"category": "finance"}),
            Document(id="travel", text="Passport", metadata={"category": ["travel", "lifestyle"]}),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    results = store.search(
        [1.0, 0.0], top_k=1, metadata_filter={"category": "travel"}
    )

    assert [result.document.id for result in results] == ["travel"]
