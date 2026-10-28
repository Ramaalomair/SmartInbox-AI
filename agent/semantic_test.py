
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

from agent.retrieval import KnowledgeRetriever


MODEL_NAME = "intfloat/multilingual-e5-small"


def main():
    """Compare semantic similarity across policy chunks."""

    print("Loading E5 embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    retriever = KnowledgeRetriever()
    chunks = retriever.chunks

    # E5 expects passage and query prefixes.
    passages = [
        "passage: " + chunk["text"]
        for chunk in chunks
    ]

    print(f"Encoding {len(passages)} policy chunks...")

    passage_embeddings = model.encode(
        passages,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    queries = [
        "What is the standard return window?",
        "My parcel never showed up. What should I do?",
        "متى أقدر أرجع المنتج بعد استلامه؟",
        "How should suspicious requests be handled?"
    ]

    for query in queries:
        query_embedding = model.encode(
            "query: " + query,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        scores = util.cos_sim(
            query_embedding,
            passage_embeddings
        )[0]

        top_k = min(3, len(chunks))
        top_results = scores.topk(k=top_k)

        print(f"\nQuery: {query}")

        for score, index in zip(
            top_results.values,
            top_results.indices
        ):
            chunk = chunks[index.item()]

            print(
                f"  {chunk['chunk_id']} | "
                f"semantic_score={score.item():.4f}"
            )
            print(
                f"  Excerpt: {chunk['text'][:160]}"
            )


if __name__ == "__main__":
    main()
