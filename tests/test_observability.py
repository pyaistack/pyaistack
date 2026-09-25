from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime

import pytest

from pyaistack import RAG, RAGConfig
from pyaistack.exceptions import ProviderError
from pyaistack.observability import InMemoryObserver, JsonlObserver, RAGEvent


class Embeddings:
    model_name = "fake-embedding"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class Chat:
    model_name = "fake-chat"

    def chat(self, messages):
        return "answer"


class BrokenChat(Chat):
    def chat(self, messages):
        raise ProviderError("provider unavailable")


def build_rag(*, observer=None, config=None, chat=None) -> RAG:
    observers = [observer] if observer is not None else None
    return RAG(
        embedding_provider=Embeddings(),
        chat_provider=chat or Chat(),
        config=config,
        observers=observers,
    )


def test_events_share_one_run_id_and_exclude_content_by_default() -> None:
    observer = InMemoryObserver()
    rag = build_rag(observer=observer)
    rag.add("secret document text", metadatas=[{"category": "finance", "secret": "value"}])
    rag.ask("secret question")

    events = observer.events
    assert {event.run_id for event in events if event.operation != "indexing"} != set()
    ask_events = events[3:]
    assert len({event.run_id for event in ask_events}) == 1
    serialized = json.dumps([event.to_dict() for event in events])
    assert "secret document text" not in serialized
    assert "secret question" not in serialized
    assert '"secret"' not in serialized
    assert all(event.duration_ms >= 0 for event in events)
    assert all(event.schema_version == 1 for event in events)
    assert all(event.timestamp.endswith("Z") for event in events)
    assert all(datetime.fromisoformat(event.timestamp.replace("Z", "+00:00")) for event in events)


def test_allowlisted_metadata_is_bounded_and_opt_in() -> None:
    observer = InMemoryObserver()
    rag = build_rag(
        observer=observer,
        config=RAGConfig(observability_metadata_keys=("category",)),
    )
    rag.add("document", metadatas=[{"category": "finance", "source": "/private/path"}])

    indexing = next(event for event in observer.events if event.operation == "indexing")
    assert indexing.details["metadata"] == {"category": ["finance"]}
    assert "/private/path" not in json.dumps(indexing.to_dict())


def test_failure_event_preserves_the_original_provider_error() -> None:
    observer = InMemoryObserver()
    rag = build_rag(observer=observer, chat=BrokenChat())
    rag.add("document")

    with pytest.raises(ProviderError, match="provider unavailable"):
        rag.ask("question")

    failures = [event for event in observer.events if event.status == "failure"]
    assert {event.operation for event in failures} == {"generation", "answer"}
    assert all(event.error_type == "ProviderError" for event in failures)


def test_observer_failure_does_not_change_rag_behavior() -> None:
    class BrokenObserver:
        def on_event(self, event) -> None:
            raise RuntimeError("observer failed")

    rag = build_rag(observer=BrokenObserver())
    rag.add("document")

    assert rag.ask("question").text == "answer"


def test_jsonl_observer_writes_sanitized_events(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    with JsonlObserver(path) as observer:
        rag = build_rag(observer=observer)
        rag.add("private text")

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records
    assert all(record["schema_version"] == 1 for record in records)
    assert all(record["timestamp"].endswith("Z") for record in records)
    assert all("private text" not in json.dumps(record) for record in records)
    with pytest.raises(ValueError, match="closed"):
        observer.on_event(
            RAGEvent(
                run_id="test",
                operation="test",
                status="success",
                duration_ms=0,
            )
        )


def test_clear_emits_removed_document_count() -> None:
    observer = InMemoryObserver()
    rag = build_rag(observer=observer)
    rag.add(["first", "second"])

    observer.clear()
    rag.clear()

    assert rag.document_count == 0
    assert len(observer.events) == 1
    event = observer.events[0]
    assert event.operation == "clear"
    assert event.status == "success"
    assert event.details == {"removed_document_count": 2}


def test_store_backend_name_is_sanitized_and_opt_in() -> None:
    observer = InMemoryObserver()
    rag = build_rag(observer=observer)
    rag.vector_store.observability_name = "postgresql"
    rag.add("document")

    indexing = next(event for event in observer.events if event.operation == "indexing")
    assert indexing.details["backend"] == "postgresql"

    rag.vector_store.observability_name = "postgresql password=secret"
    observer.clear()
    rag.clear()
    assert "backend" not in observer.events[0].details
