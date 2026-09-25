# Retrieval comparison

Run from the checkout after installing development dependencies:

```bash
python benchmarks/retrieval.py
python benchmarks/retrieval.py --model embeddinggemma --repeats 5
python benchmarks/retrieval.py --fixture
```

The 24 authored documents live in `examples/hybrid_knowledge/`; 12 manually
labeled questions cover identifiers, product names, and paraphrases.
The database is temporary and deleted on exit. No user database is changed.

The report includes model, Python/SQLite/platform, dimension, configuration,
Recall@5, MRR@5, query-level rankings, median/p95 retrieval time, and peak Python
allocations. Embeddings and warmup/FTS migration are excluded from retrieval
timings; their inclusion would answer a different latency question.
Python allocation peaks are not process RSS or Ollama model memory.

Fixture embeddings use deterministic token hashing and cannot demonstrate
semantic search quality. Real-model results on this small authored corpus are
smoke evidence only. Expand with independently labeled application questions
before making broader claims; retain ties and regressions in published reports.

For a capacity study, repeat on 1k/10k/50k representative chunks and include
OS-level memory, concurrency, corpus construction, hardware, and model details.
The supplied small benchmark does not establish those capacity levels.
