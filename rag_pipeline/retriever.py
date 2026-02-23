"""
RAG Retriever Module
Embeds text chunks using sentence-transformers and retrieves the most relevant ones via FAISS.
"""
import os
import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from rag_pipeline.chunking import get_all_chunks


class RAGRetriever:
    """
    Phase 9: RAG Retriever
    - Chunking vehicle maintenance data
    - Embedding with sentence-transformers
    - Vector Storage (FAISS)
    - Semantic retrieval
    """

    MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, lightweight embedding model

    def __init__(self, index_dir: str | None = None):
        if not HAS_DEPS:
            print("Warning: faiss-cpu or sentence-transformers not installed. RAG retriever running in stub mode.")
            self.chunks = []
            self.index = None
            self.model = None
            return

        if index_dir is None:
            index_dir = os.path.dirname(os.path.abspath(__file__))

        self.index_path = os.path.join(index_dir, "index.faiss")
        self.chunks_path = os.path.join(index_dir, "chunks.npy")

        print(f"Loading embedding model: {self.MODEL_NAME} ...")
        self.model = SentenceTransformer(self.MODEL_NAME)

        # Load or build the index
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            print("Loading existing FAISS index ...")
            self.index = faiss.read_index(self.index_path)
            self.chunks = list(np.load(self.chunks_path, allow_pickle=True))
            print(f"Loaded index with {self.index.ntotal} vectors, {len(self.chunks)} chunks")
        else:
            print("Building FAISS index from scratch ...")
            self._build_index()

    def _build_index(self):
        """Build the FAISS index from all available chunks."""
        self.chunks = get_all_chunks()
        if not self.chunks:
            print("Warning: No chunks found to index.")
            self.index = None
            return

        print(f"Embedding {len(self.chunks)} chunks ...")
        embeddings = self.model.encode(self.chunks, show_progress_bar=True, batch_size=64)
        embeddings = np.array(embeddings, dtype="float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build IndexFlatIP (inner product on normalized vectors = cosine similarity)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        # Save
        faiss.write_index(self.index, self.index_path)
        np.save(self.chunks_path, np.array(self.chunks, dtype=object))
        print(f"FAISS index saved: {self.index.ntotal} vectors, dim={dimension}")

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Retrieve the top-k most relevant chunks for a query."""
        if not self.index or not self.model:
            return ["RAG index not available. Please ensure dependencies are installed."]

        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding, dtype="float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results


if __name__ == "__main__":
    retriever = RAGRetriever()
    query = "When should I change the oil?"
    results = retriever.retrieve(query, k=3)
    print(f"\nQuery: {query}")
    for i, chunk in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(chunk[:300])
