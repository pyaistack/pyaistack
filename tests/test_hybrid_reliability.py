"""Regression coverage for bounded context and opt-in hybrid retrieval."""

import sqlite3

import pytest

from pyaistack import RAG, Document, RAGConfig, SearchResult
from pyaistack.exceptions import ConfigurationError, MetadataFactoryError, VectorStoreError
from pyaistack.loaders import CSVMetadataFactory
from pyaistack.loaders.llm_metadata import _normalize_value
from pyaistack.rag.context import build_context
from pyaistack.vectorstores import InMemoryVectorStore, SQLiteVectorStore
from pyaistack.vectorstores._metadata import metadata_matches
from pyaistack.vectorstores._vectors import normalize


@pytest.mark.parametrize(
    "field,value",
    [
        ("top_k", 1.5),
        ("top_k", True),
        ("min_score", float("nan")),
        ("max_context_chars", False),
        ("candidate_k", 0),
        ("retrieval_mode", "typo"),
    ],
)
def test_invalid_config(field, value):
    with pytest.raises(ConfigurationError):
        RAGConfig(**{field: value})


def test_context_exact_budget_and_excerpts():
    sources = [SearchResult(Document(str(i), "a" * 40), 1.0) for i in range(3)]
    text, included = build_context(sources, 100)
    assert len(text) <= 100
    assert len(included) == 2
    assert len(included[-1].document.text) < 40
    assert build_context(sources, 1) == ("", ())


class Embeddings:
    model_name = "test"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def __bool__(self):
        return False


class Chat:
    model_name = "test"

    def chat(self, messages):
        return "answer"

    def __bool__(self):
        return False


def test_false_providers_and_context_sources():
    rag = RAG(
        embedding_provider=Embeddings(),
        chat_provider=Chat(),
        config=RAGConfig(max_context_chars=50, include_sources=True),
    )
    rag.add(["a" * 100, "b" * 100])
    answer = rag.ask("question")
    assert len(answer.sources) == 2
    assert len(answer.context_sources) == 1
    assert len(answer.context_sources[0].document.text) == 40


def test_hybrid_requires_capability():
    with pytest.raises(ConfigurationError, match="hybrid"):
        RAG(
            embedding_provider=Embeddings(),
            chat_provider=Chat(),
            config=RAGConfig(retrieval_mode="hybrid"),
        )


def test_fts_migration_persistence_clear_and_sync(tmp_path):
    path = tmp_path / "index.db"
    with SQLiteVectorStore(path) as store:
        store.add(
            [Document("a", "general finance"), Document("z", "FIN-042 purchase rules")],
            [[1, 0], [0, 1]],
        )
        assert store.search([1, 0], top_k=1)[0].document.id == "a"
        hybrid = store.hybrid_search("FIN-042", [1, 0], top_k=2)
        assert hybrid[0].document.id == "z"
        assert hybrid[0].score == 0
        assert hybrid[0].ranking_score > 0
    with SQLiteVectorStore(path) as store:
        assert store.hybrid_search("FIN-042", [1, 0], top_k=1)[0].document.id == "z"
        store.add([Document("new", "FIN-043 approvals")], [[1, 0]])
        assert store.hybrid_search("FIN-043", [1, 0], top_k=1)[0].document.id == "new"
        store.clear()
        store.add([Document("after", "new policy")], [[1, 0]])
        assert [r.document.id for r in store.hybrid_search("FIN-042", [1, 0], top_k=5)] == ["after"]


@pytest.mark.parametrize(
    "metadata_filter",
    [
        {},
        {"x": None},
        {"x": ["a", "b"]},
        {"x": "a"},
        {"x": []},
        {"missing": None},
        {"x": {"n": 1}},
        {"x": 1},
        {"x": True},
    ],
)
def test_filter_parity(tmp_path, metadata_filter):
    metadata = [
        {},
        {"x": None},
        {"x": ["a", "b"]},
        {"x": "a"},
        {"x": {"n": 1}},
        {"x": 1},
        {"x": True},
    ]
    docs = [Document(str(i), "test policy", m) for i, m in enumerate(metadata)]
    expected = {d.id for d in docs if metadata_matches(d.metadata, metadata_filter)}
    for store in (InMemoryVectorStore(), SQLiteVectorStore(tmp_path / "index.db")):
        store.add(docs, [[1, 0]] * len(docs))
        assert {
            r.document.id for r in store.search([1, 0], top_k=20, metadata_filter=metadata_filter)
        } == expected
        if isinstance(store, SQLiteVectorStore):
            assert {
                r.document.id
                for r in store.hybrid_search(
                    "policy", [1, 0], top_k=20, metadata_filter=metadata_filter
                )
            } == expected
            store.close()


def test_filtered_corrupt_vector_never_deserialized(tmp_path):
    with SQLiteVectorStore(tmp_path / "index.db") as store:
        store.add(
            [Document("a", "policy", {"cat": "yes"}), Document("b", "other", {"cat": "no"})],
            [[1, 0], [1, 0]],
        )
        store._connection.execute("UPDATE pyaistack_vectors SET vector=x'01' WHERE document_id='b'")
        store._connection.commit()
        assert len(store.search([1, 0], top_k=2, metadata_filter={"cat": "yes"})) == 1
        with pytest.raises(VectorStoreError):
            store.search([1, 0], top_k=2)


def test_model_rebinding_and_atomic_duplicate_failure(tmp_path):
    with SQLiteVectorStore(tmp_path / "index.db") as store:
        rag = RAG(embedding_provider=Embeddings(), chat_provider=Chat(), vector_store=store)
        rag.add("first")
        rag.clear()
        rag.add("second")
        with pytest.raises(VectorStoreError, match="model changed"):
            store.set_embedding_model("other")
        with pytest.raises(VectorStoreError):
            store.add([Document("dup", "a"), Document("dup", "b")], [[1, 0], [1, 0]])
        assert len(store) == 1
    store.close()
    with pytest.raises(VectorStoreError, match="closed"):
        len(store)


def test_fts_unavailable_rolls_back(tmp_path):
    with SQLiteVectorStore(tmp_path / "index.db") as store:
        store.add([Document("a", "policy")], [[1, 0]])
        store._connection.set_authorizer(
            lambda action, *args: (
                sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_CREATE_VTABLE else sqlite3.SQLITE_OK
            )
        )
        with pytest.raises(VectorStoreError):
            store.hybrid_search("policy", [1, 0], top_k=1)
        store._connection.set_authorizer(None)
        assert len(store.search([1, 0], top_k=1)) == 1
        assert len(store.hybrid_search("policy", [1, 0], top_k=1)) == 1


def test_empty_and_literal_queries(tmp_path):
    with SQLiteVectorStore(tmp_path / "index.db") as store:
        store.add([Document("a", "policy")], [[1, 0]])
        assert len(store.hybrid_search('OR " () : *', [1, 0], top_k=2)) == 1
        assert store.hybrid_search("!!!", [1, 0], top_k=2)[0].ranking_score is None


def test_metadata_normalization_preserves_values():
    assert _normalize_value("topic", "वित्त") == "वित्त"
    assert _normalize_value("created_at", "2026-09-12") == "2026-09-12"
    assert _normalize_value("version", "v1.2") == "v1.2"
    assert _normalize_value("title", "Hello, World") == "Hello, World"


@pytest.mark.parametrize(
    "row",
    [
        'a.txt,"[]"',
        'a.txt,"{""x"": NaN}"',
        'a.txt,"{""source"": ""bad""}"',
        '../a.txt,"{}"',
        'a.txt,"{}",extra',
        'a.txt,"{}"\na.txt,"{}"',
    ],
)
def test_csv_rejects_bad_manifest(tmp_path, row):
    path = tmp_path / "metadata.csv"
    path.write_text("source,metadata_json\n" + row, encoding="utf-8")
    with pytest.raises(MetadataFactoryError):
        CSVMetadataFactory(path, root=tmp_path)


def test_normalization_handles_large_and_bad_vectors():
    assert normalize([1e308, 1e308])[0] == pytest.approx(2**-0.5)
    for vector in ([], [0, 0], [float("nan")], ["bad"]):
        with pytest.raises(VectorStoreError):
            normalize(vector)
