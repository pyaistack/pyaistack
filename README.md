<p align="center">
  <img src="docs/assets/logo-v1.png" alt="PyAIStack" width="720">
</p>

<h1 align="center">PyAIStack</h1>

<p align="center">
  ⚡ A lightweight, modular microframework for complete AI applications.
</p>

PyAIStack helps you build RAG, agents, agentic tools, AI features, and provider integrations faster. This deliberately small first version implements the production-oriented RAG foundation, using Ollama and Gemma models by default.

## ✨ Start simple

```python
from pyaistack import RAG

rag = RAG(
    embedding_model="embeddinggemma",
    llm_model="gemma3:4b",
)

rag.add([
    "AWS Lambda is a serverless compute service.",
    "Amazon S3 is an object storage service.",
])

answer = rag.ask("Which service runs code without managing servers?")
print(answer.text)
```

## ✨ What v0.1 provides

- `RAG` facade for indexing, retrieval, context construction and generation.
- Ollama embedding provider using `embeddinggemma` by default.
- Ollama chat provider using `gemma3:4b` by default.
- Embedding model can be changed with one constructor argument.
- Provider interfaces so Ollama can later be replaced with OpenAI, Bedrock, etc.
- Thread-safe in-memory vector store with cosine similarity.
- Vector dimension validation to prevent accidental mixed embedding models.
- Metadata attached to each document and returned with sources.
- Explicit source objects in every generated answer.
- Bounded context size.
- Basic retry/backoff and provider-specific errors.
- Dependency injection for deterministic unit tests.
- Type hints and small modules rather than one large class.

## 🧠 Requirements

- Python 3.10+
- Ollama running locally
- Ollama version compatible with `embeddinggemma` (the Ollama model page currently specifies v0.11.10+)

Pull the default models:

```bash
ollama pull embeddinggemma
ollama pull gemma3:4b
```

## 🚀 Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install pyaistack
```

For development:

```bash
pip install -e ".[dev]"
```

## ▶️ Run

```bash
python examples/basic.py
```

### How it works

```text
documents
   │
   ▼
OllamaEmbeddingProvider
   │
   ▼
InMemoryVectorStore

question
   │
   ▼
embedding
   │
   ▼
cosine retrieval
   │
   ▼
Top-K sources
   │
   ▼
context builder
   │
   ▼
OllamaChatProvider
   │
   ▼
RAGAnswer(text + sources)
```

## 🔧 Change the embedding model

Only configuration changes:

```python
rag = RAG(
    embedding_model="qwen3-embedding:0.6b",
    llm_model="gemma3:4b",
)
```

or:

```python
rag = RAG(
    embedding_model="nomic-embed-text",
    llm_model="gemma3:4b",
)
```

> Note: Do not change embedding models while an index contains vectors. Different models may produce different vector dimensions and, more importantly, incompatible vector spaces. Clear/rebuild the index when changing the embedding model.

## 🌐 Configure Ollama host

```python
rag = RAG(
    embedding_model="embeddinggemma",
    llm_model="gemma3:4b",
    ollama_host="http://192.168.1.20:11434",
)
```

## 🔎 Access retrieval results without generating

```python
results = rag.search("serverless compute", top_k=3)

for result in results:
    print(result.score)
    print(result.document.text)
    print(result.document.metadata)
```

## 🏷️ Add metadata

```python
rag.add(
    ["Leave policy text", "Travel policy text"],
    metadatas=[
        {"file": "leave-policy.pdf", "page": 3},
        {"file": "travel-policy.pdf", "page": 7},
    ],
)
```

## ⚙️ Configure retrieval

```python
from pyaistack import RAG, RAGConfig

rag = RAG(
    config=RAGConfig(
        top_k=5,
        min_score=0.25,
        max_context_chars=20_000,
    )
)
```

## 💬 Friendly errors

Use `format_error` at your application entry point to show the relevant application line without PyAIStack implementation frames:

```python
from pyaistack import RAG, format_error
from pyaistack.exceptions import ProviderError

rag = RAG()

try:
    rag.add(["A document to index."])
except ProviderError as error:
    print(format_error(error))
```

For example, a missing embedding model is displayed as:

```text
Traceback (most recent call last):
  File "/path/to/app.py", line 8, in <module>
    rag.add(["A document to index."])
pyaistack.exceptions.ProviderError: Model "embeddinggemma" is not available.
Run: ollama pull embeddinggemma
```

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE).
