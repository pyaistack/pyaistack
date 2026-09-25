# Retrieval benchmark — recorded development run

This records the live Ollama run performed during 0.2.2 development, before the
subsequent scalar SQL-filter optimization. It is not a benchmark of the final
release checkout. No accuracy improvement was observed on this small corpus.

## Method

- Embedding model: `embeddinggemma`, 768 dimensions.
- Corpus: 24 authored documents; 12 manually labeled questions.
- Configuration: `top_k=5`, `candidate_k=20`, no minimum cosine threshold.
- Five measured repetitions after warmup; query/document embeddings computed once.
- Environment: Linux x86_64, Python 3.11.16, SQLite 3.37.2.
- CPU model, RAM and GPU details were not captured; hardware comparisons are unsupported.
- Combined embedding time: 4.88 seconds.
- Retrieval timings exclude embeddings and initial FTS migration, with Python allocation tracing enabled.

## Observations

| Metric | Vector | Hybrid |
| --- | ---: | ---: |
| Recall@5 | 1.000 | 1.000 |
| MRR@5 | 1.000 | 1.000 |
| Median retrieval time | 22.74 ms | 30.22 ms |
| p95 retrieval time | 24.51 ms | 41.32 ms |
| Peak traced Python allocations | 85,809 bytes | 116,321 bytes |

Both modes retrieved the labeled answer first for all 12 questions. Hybrid
changed some lower-ranked results, but added latency and Python allocations.
These memory figures exclude native SQLite allocations, total process RSS,
and Ollama model memory.

This corpus is too small to establish general accuracy or large-index capacity.
Do not advertise improved accuracy, speed, or a production capacity number
from these measurements.

## Reproduce

```bash
python benchmarks/retrieval.py --model embeddinggemma --repeats 5
```

The script prints environment details, metrics, and query-level rankings.
Use independently labeled application questions for a broader evaluation and
report ties and regressions as well as gains. See [benchmark methodology](README.md).
