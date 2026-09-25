"""Opt in to sanitized RAG events through logging and an application-owned JSONL file."""

import logging

from pyaistack import RAG
from pyaistack.observability import JsonlObserver, LoggingObserver

logging.basicConfig(level=logging.INFO, format="%(message)s")

# PyAIStack creates no observability events unless observers are passed here.
# The JSONL file is selected and owned by this application.
with JsonlObserver("rag_events.jsonl") as file_observer:
    rag = RAG(observers=[LoggingObserver(), file_observer])
    rag.add(
        [
            "AWS Lambda runs code without managing servers.",
            "Amazon S3 stores files and objects.",
        ],
        metadatas=[{"category": "cloud"}, {"category": "storage"}],
    )
    answer = rag.ask("Which service runs code without managing servers?")
    print(answer.text)

print("Sanitized events were written to rag_events.jsonl")
