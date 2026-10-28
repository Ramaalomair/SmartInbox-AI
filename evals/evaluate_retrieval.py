
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from agent.retrieval import KnowledgeRetriever
from agent.hybrid_retrieval import HybridKnowledgeRetriever


ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "evals" / "retrieval_cases.json"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


def load_cases():
    """Load labeled retrieval evaluation cases."""
    with CASES_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def semantic_search(retriever, model, embeddings, query, top_k=3):
    """Search policy chunks using semantic embeddings."""

    query_embedding = model.encode(
        "query: " + query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    scores = embeddings @ query_embedding
    ranked_indices = np.argsort(-scores, kind="stable")[:top_k]

    return [
        {
            "source_id": retriever.chunks[index]["source_id"],
            "chunk_id": retriever.chunks[index]["chunk_id"],
            "score": float(scores[index])
        }
        for index in ranked_indices
    ]


def evaluate_method(name, search_fn, cases):
    """Calculate Hit@3 and MRR for a retrieval method."""

    hit_count = 0
    reciprocal_rank_sum = 0.0
    failed_cases = []

    for case in cases:
        results = search_fn(case["query"])

        expected_sources = set(case["expected_sources"])
        first_correct_rank = None

        for rank, result in enumerate(results, start=1):
            if result["source_id"] in expected_sources:
                first_correct_rank = rank
                break

        if first_correct_rank is not None:
            hit_count += 1
            reciprocal_rank_sum += 1.0 / first_correct_rank
        else:
            failed_cases.append(case["id"])

    total = len(cases)

    return {
        "method": name,
        "hit_at_3": hit_count / total,
        "mrr_at_3": reciprocal_rank_sum / total,
        "failed_cases": failed_cases
    }


def main():
    """Compare TF-IDF, semantic E5, and hybrid RRF retrieval."""

    cases = load_cases()

    if not cases:
        raise ValueError("No retrieval evaluation cases found.")

    print(f"Loaded {len(cases)} evaluation cases.")

    print("\nLoading TF-IDF retriever...")
    lexical = KnowledgeRetriever()

    print("Loading Hybrid retriever and E5 model...")
    hybrid = HybridKnowledgeRetriever()

    # Reuse the hybrid retriever's loaded embedding model and vectors.
    model = hybrid.embedding_model
    embeddings = hybrid.embeddings

    methods = [
        (
            "TF-IDF",
            lambda query: lexical.search(query, top_k=3)
        ),
        (
            "Semantic E5",
            lambda query: semantic_search(
                hybrid,
                model,
                embeddings,
                query,
                top_k=3
            )
        ),
        (
            "Hybrid RRF",
            lambda query: hybrid.search(query, top_k=3)
        )
    ]

    print("\n===== Retrieval Evaluation Results =====")

    for name, search_fn in methods:
        result = evaluate_method(
            name,
            search_fn,
            cases
        )

        print(f"\nMethod: {result['method']}")
        print(f"Hit@3: {result['hit_at_3']:.2%}")
        print(f"MRR@3: {result['mrr_at_3']:.4f}")
        print(
            "Failed cases:",
            result["failed_cases"] or "None"
        )


if __name__ == "__main__":
    main()
