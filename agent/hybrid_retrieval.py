
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from agent.retrieval import KnowledgeRetriever


EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


class HybridKnowledgeRetriever(KnowledgeRetriever):
    """Combine lexical and semantic retrieval using RRF."""

    def __init__(self, knowledge_dir=None):
        super().__init__(knowledge_dir)

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        passages = [
            "passage: " + chunk["text"]
            for chunk in self.chunks
        ]

        self.embeddings = self.embedding_model.encode(
            passages,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    def search(self, query, top_k=3):
        """Retrieve policy chunks using hybrid ranking."""

        if not query.strip():
            return []

        # Lexical scores using TF-IDF.
        query_vector = self.vectorizer.transform([query])

        lexical_scores = cosine_similarity(
            query_vector,
            self.matrix
        )[0]

        # Semantic scores using multilingual E5.
        query_embedding = self.embedding_model.encode(
            "query: " + query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        semantic_scores = self.embeddings @ query_embedding

        # Rank chunks independently for each retrieval method.
        lexical_ranking = np.argsort(
            -lexical_scores,
            kind="stable"
        )

        semantic_ranking = np.argsort(
            -semantic_scores,
            kind="stable"
        )

        # Reciprocal Rank Fusion.
        rrf_k = 60
        fused_scores = np.zeros(len(self.chunks))

        for rank, index in enumerate(lexical_ranking, start=1):
            if lexical_scores[index] > 0:
                fused_scores[index] += 1.0 / (rrf_k + rank)

        for rank, index in enumerate(semantic_ranking, start=1):
            fused_scores[index] += 1.0 / (rrf_k + rank)

        ranked_indices = np.argsort(
            -fused_scores,
            kind="stable"
        )

        results = []

        for index in ranked_indices[:max(1, top_k)]:
            chunk = self.chunks[index]

            results.append({
                "source_id": chunk["source_id"],
                "chunk_id": chunk["chunk_id"],
                "excerpt": chunk["text"],
                "score": round(float(fused_scores[index]), 6)
            })

        return results


if __name__ == "__main__":
    retriever = HybridKnowledgeRetriever()

    queries = [
        "What is the standard return window?",
        "My parcel never showed up. What should I do?",
        "متى أقدر أرجع المنتج بعد استلامه؟",
        "How should suspicious requests be handled?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")

        for result in retriever.search(query, top_k=3):
            print(
                f"  {result['chunk_id']} | "
                f"rrf_score={result['score']}"
            )
            print(
                f"  Excerpt: {result['excerpt'][:160]}"
            )
