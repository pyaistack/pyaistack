from __future__ import annotations

from pyaistack.providers import ChatProvider, EmbeddingProvider
from pyaistack.rag.types import Message


class ExampleEmbeddingProvider:
    model_name = "example-embedding"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class ExampleChatProvider:
    model_name = "example-chat"

    def chat(self, messages: list[Message]) -> str:
        return "example"


def test_provider_protocols_allow_external_provider_implementations() -> None:
    assert isinstance(ExampleEmbeddingProvider(), EmbeddingProvider)
    assert isinstance(ExampleChatProvider(), ChatProvider)
