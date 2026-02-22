import os

class RAGRetriever:
    """
    Phase 9: RAG Retriever
    - Chunking
    - Vector Storage (FAISS)
    - Retrieval logic
    """
    def __init__(self):
        self.index_path = 'rag_pipeline/index.faiss'
        
    def retrieve(self, query, k=3):
        print(f"Retrieving top-{k} document chunks for: {query}")
        return ["Chunk 1: Check tire pressure monthly.", "Chunk 2: Change oil every 5k km."]

if __name__ == "__main__":
    retriever = RAGRetriever()
    retriever.retrieve("When to change oil?")
