"""Ask a persisted index with explicitly selected Ollama providers."""

import argparse

from pyaistack import RAG, RAGConfig
from pyaistack.providers import OllamaChatProvider, OllamaEmbeddingProvider
from pyaistack.vectorstores import SQLiteVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="What does policy FIN-042 require?")
    parser.add_argument("--db", default="text_knowledge.db")
    parser.add_argument("--host", default="http://localhost:11434")
    args = parser.parse_args()

    # The embedding model must match the model used when this SQLite index was created.
    embedding_provider = OllamaEmbeddingProvider(model="embeddinggemma", host=args.host)

    # Chat and embedding models can be selected independently.
    chat_provider = OllamaChatProvider(model="gemma3:4b", host=args.host)

    with SQLiteVectorStore(args.db) as store:
        rag = RAG(
            embedding_provider=embedding_provider,
            chat_provider=chat_provider,
            vector_store=store,
            config=RAGConfig(top_k=3, min_score=0.15),
        )
        print(rag.ask(args.question).text)


if __name__ == "__main__":
    main()
