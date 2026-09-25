"""Use explicit Ollama providers and model names with an in-memory RAG index."""

from pyaistack import RAG
from pyaistack.providers import OllamaChatProvider, OllamaEmbeddingProvider

# Configure the embedding model separately from the chat model.
# These defaults are equivalent to calling RAG() with no providers.
embedding_provider = OllamaEmbeddingProvider(
    model="embeddinggemma",
    host="http://localhost:11434",
)
chat_provider = OllamaChatProvider(
    model="gemma3:4b",
    host="http://localhost:11434",
)

rag = RAG(
    embedding_provider=embedding_provider,
    chat_provider=chat_provider,
)
rag.add(
    [
        "AWS Lambda runs code without managing servers.",
        "Amazon S3 stores files and objects.",
    ],
    metadatas=[{"source": "aws-lambda"}, {"source": "aws-s3"}],
)

answer = rag.ask("Which AWS service runs code without managing servers?")
print(answer.text)
