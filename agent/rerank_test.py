
from sentence_transformers import CrossEncoder

from agent.hybrid_retrieval import HybridKnowledgeRetriever


RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def main():
    """Evaluate cross-encoder reranking on hybrid candidates."""

    retriever = HybridKnowledgeRetriever()

    print("Loading reranker...")
    reranker = CrossEncoder(RERANK_MODEL)

    queries = [
        "My parcel never showed up. What should I do?",
        "What is the standard return window?",
        "How should suspicious requests be handled?",
        "متى أقدر أرجع المنتج بعد استلامه؟"
    ]

    for query in queries:
        candidates = retriever.search(query, top_k=10)

        pairs = [
            (query, candidate["excerpt"])
            for candidate in candidates
        ]

        scores = reranker.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True
        )

        print(f"\nQuery: {query}")

        for candidate, score in ranked[:3]:
            print(
                f"  {candidate['chunk_id']} | "
                f"rerank_score={float(score):.4f}"
            )


if __name__ == "__main__":
    main()
