"""Compare exact vector retrieval and hybrid search on a fixed authored corpus.

Default uses real Ollama embeddings. --fixture uses token hashing solely to
exercise benchmark plumbing and must not be cited as semantic-model evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sqlite3
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path

from pyaistack import Document
from pyaistack.providers import OllamaEmbeddingProvider
from pyaistack.vectorstores import SQLiteVectorStore

ROOT = Path(__file__).resolve().parents[1]


class FixtureEmbedding:
    model_name = "fixture-token-hash-not-semantic"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * 128
            for token in re.findall(r"\w+", text.casefold()):
                bucket = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % 128
                vector[bucket] += 1
            vectors.append(vector)
        return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--model", default="embeddinggemma")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    provider = (
        FixtureEmbedding()
        if args.fixture
        else OllamaEmbeddingProvider(
            model=args.model,
            host=args.host,
        )
    )
    documents = [
        Document(path.stem, path.read_text(encoding="utf-8"), {"source": path.name})
        for path in sorted((ROOT / "examples/hybrid_knowledge").glob("*.txt"))
    ]
    questions = json.loads((ROOT / "benchmarks/retrieval_queries.json").read_text())
    started = time.perf_counter()
    vectors = provider.embed([d.text for d in documents])
    query_vectors = provider.embed([q["query"] for q in questions])
    embedding_seconds = time.perf_counter() - started
    report = {
        "model": provider.model_name,
        "fixture": args.fixture,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "documents": len(documents),
        "queries": len(questions),
        "dimension": len(vectors[0]),
        "top_k": 5,
        "candidate_k": 20,
        "repeats": args.repeats,
        "embedding_seconds": embedding_seconds,
        "notes": "Small authored corpus, not a general accuracy or capacity claim. "
        "Latency excludes embeddings and warmup; peak is Python allocations, not total RSS.",
    }
    with tempfile.TemporaryDirectory(prefix="pyaistack-benchmark-") as directory:
        with SQLiteVectorStore(Path(directory) / "benchmark.db") as store:
            store.set_embedding_model(provider.model_name)
            store.add(documents, vectors)
            for mode in ("vector", "hybrid"):

                def search(index: int, mode: str = mode):
                    if mode == "hybrid":
                        return store.hybrid_search(
                            questions[index]["query"], query_vectors[index], top_k=5, candidate_k=20
                        )
                    return store.search(query_vectors[index], top_k=5)

                for index in range(len(questions)):
                    search(index)
                durations = []
                recalls, reciprocal_ranks, details = [], [], []
                tracemalloc.start()
                for repeat in range(args.repeats):
                    for index, question in enumerate(questions):
                        start = time.perf_counter()
                        results = search(index)
                        durations.append((time.perf_counter() - start) * 1000)
                        if repeat == 0:
                            ids = [r.document.id for r in results]
                            relevant = set(question["relevant"])
                            recalls.append(len(relevant.intersection(ids)) / len(relevant))
                            reciprocal_ranks.append(
                                next(
                                    (
                                        1 / rank
                                        for rank, key in enumerate(ids, 1)
                                        if key in relevant
                                    ),
                                    0,
                                )
                            )
                            details.append({"query": question["query"], "results": ids})
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                report[mode] = {
                    "recall_at_5": statistics.mean(recalls),
                    "mrr_at_5": statistics.mean(reciprocal_ranks),
                    "median_ms": statistics.median(durations),
                    "p95_ms": sorted(durations)[max(0, int(len(durations) * 0.95) - 1)],
                    "peak_python_bytes": peak,
                    "queries": details,
                }
    report["recall_improved"] = report["hybrid"]["recall_at_5"] > report["vector"]["recall_at_5"]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
