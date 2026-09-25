from __future__ import annotations

import os
from uuid import uuid4

import pytest

from pyaistack.exceptions import DependencyError, VectorStoreError
from pyaistack.rag.types import Document
from pyaistack.vectorstores import PgVectorSchema, PgVectorStore

DSN = os.environ.get("PYAISTACK_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="PYAISTACK_TEST_POSTGRES_DSN is not set")


def build_store(*, collection: str, dimensions: int = 2) -> PgVectorStore:
    try:
        return PgVectorStore(
            dsn=DSN,
            dimensions=dimensions,
            collection=collection,
            schema=PgVectorSchema(
                schema_name="pyaistack_test",
                table_name="documents_2d",
            ),
            initialize=True,
            search_mode="exact",
        )
    except DependencyError as error:
        pytest.skip(str(error))


def test_pgvector_persistence_filters_hybrid_and_collection_isolation() -> None:
    first_collection = f"first-{uuid4()}"
    second_collection = f"second-{uuid4()}"
    with build_store(collection=first_collection) as first:
        first.set_embedding_model("test-embedding")
        first.add(
            [
                Document("finance", "FIN-042 budget policy", {"category": "finance"}),
                Document("travel", "passport guidance", {"category": ["travel", "guide"]}),
            ],
            [[1, 0], [0, 1]],
        )
        assert first.search([1, 0], top_k=1)[0].document.id == "finance"
        assert first.search(
            [1, 0], top_k=2, metadata_filter={"category": "travel"}
        )[0].document.id == "travel"
        assert first.hybrid_search("FIN-042", [0, 1], top_k=1)[0].document.id == "finance"
        with pytest.raises(VectorStoreError):
            first.add(
                [Document("duplicate", "first"), Document("duplicate", "second")],
                [[1, 0], [1, 0]],
            )
        assert len(first) == 2

    with build_store(collection=first_collection) as reopened:
        reopened.set_embedding_model("test-embedding")
        assert len(reopened) == 2

    with build_store(collection=second_collection) as second:
        second.set_embedding_model("test-embedding")
        assert len(second) == 0
        second.add([Document("other", "other collection")], [[1, 0]])
        second.clear()
        assert len(second) == 0

    with build_store(collection=first_collection) as first:
        first.set_embedding_model("test-embedding")
        assert len(first) == 2
        first.clear()
        assert len(first) == 0
        with pytest.raises(VectorStoreError, match="different collection"):
            first.set_embedding_model("different-model")
