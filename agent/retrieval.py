
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeRetriever:
    """Retrieve relevant policy chunks using TF-IDF."""

    def __init__(self, knowledge_dir=None):
        if knowledge_dir is None:
            knowledge_dir = (
                Path(__file__).resolve().parent.parent
                / "knowledge_base"
            )

        self.knowledge_dir = Path(knowledge_dir)
        self.chunks = []

        self._load_documents()

        if not self.chunks:
            raise ValueError("No knowledge base chunks were found.")

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2)
        )

        self.matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in self.chunks]
        )

    def _load_documents(self):
        """Load Markdown files and split them into chunks."""

        for file_path in sorted(self.knowledge_dir.glob("*.md")):
            content = file_path.read_text(encoding="utf-8")

            chunks = self._chunk_document(content)

            for index, chunk_text in enumerate(chunks, start=1):
                self.chunks.append({
                    "source_id": file_path.name,
                    "chunk_id": f"{file_path.stem}#chunk-{index}",
                    "text": chunk_text
                })

    @staticmethod
    def _chunk_document(text, max_chars=700):
        """Split documents into chunks at word boundaries."""

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text)
            if paragraph.strip()
        ]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # Save the current chunk before exceeding the limit.
            if (
                current_chunk
                and len(current_chunk) + len(paragraph) + 2 > max_chars
            ):
                chunks.append(current_chunk)
                current_chunk = ""

            if len(paragraph) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # Split long paragraphs at word boundaries.
                words = paragraph.split()
                piece = ""

                for word in words:
                    candidate = f"{piece} {word}".strip()

                    if piece and len(candidate) > max_chars:
                        chunks.append(piece)
                        piece = word
                    else:
                        piece = candidate

                if piece:
                    chunks.append(piece)

                continue

            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def search(self, query, top_k=3):
        """Return the most relevant policy chunks using TF-IDF."""

        if not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.matrix
        )[0]

        ranked_indices = scores.argsort()[::-1]

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score <= 0:
                continue

            chunk = self.chunks[index]

            results.append({
                "source_id": chunk["source_id"],
                "chunk_id": chunk["chunk_id"],
                "excerpt": chunk["text"],
                "score": round(score, 4)
            })

            if len(results) >= max(1, top_k):
                break

        return results


if __name__ == "__main__":
    retriever = KnowledgeRetriever()

    print("Total chunks:", len(retriever.chunks))

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
                f"score={result['score']}"
            )
            print(
                f"  Excerpt: {result['excerpt'][:160]}"
            )
