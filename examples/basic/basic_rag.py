"""Start here: index two texts in memory and ask one grounded question."""

from pyaistack import RAG

# Ollama and the default local models are used when no providers are supplied.
rag = RAG()
rag.add(
    [
        "Water plants when the top soil feels dry.",
        "Compost improves soil structure and adds nutrients.",
    ],
    metadatas=[{"source": "plant-care"}, {"source": "composting"}],
)

answer = rag.ask("When should I water plants?")
print(answer.text)
