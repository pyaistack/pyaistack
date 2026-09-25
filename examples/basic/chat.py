"""Run a small interactive RAG chat using local Ollama models."""

import logging

from pyaistack import RAG, format_error
from pyaistack.exceptions import ProviderError

logging.basicConfig(level=logging.WARNING)

rag = RAG()


def prepare_index() -> bool:
    try:
        rag.add(
            [
                (
                    "AWS Lambda is a serverless compute service that runs Python code "
                    "without managing servers."
                ),
                (
                    "Amazon S3 is an object storage service used to store files, documents, "
                    "images, and other data."
                ),
                (
                    "Amazon Bedrock provides managed access to foundation models for building "
                    "generative AI applications."
                ),
            ],
            metadatas=[
                {"source": "aws-lambda"},
                {"source": "aws-s3"},
                {"source": "aws-bedrock"},
            ],
        )
    except ProviderError as exc:
        print(f"\n{format_error(exc)}\n")
        return False
    return True


if not prepare_index():
    raise SystemExit(1)

print("\nRAG is ready.")
print("Ask questions below. Type 'exit', 'quit', or 'q' to stop.\n")

try:
    while True:
        question = input("Question: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("Exiting...")
            break

        try:
            answer = rag.ask(question)

            print("\nAnswer")
            print("------")
            print(answer.text)

            print("\nRetrieved chunks")
            print("----------------")
            for source in answer.sources:
                print(f"{source.score:.4f}  {source.document.metadata}")

            print()

        except ProviderError as exc:
            print(f"\n{format_error(exc)}\n")

except KeyboardInterrupt:
    print("\nExiting...")
