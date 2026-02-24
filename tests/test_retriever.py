"""
Unit tests for the RAG Retriever module.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestRAGRetrieverInit:
    """Tests for RAGRetriever initialization."""

    def test_init_without_deps(self):
        """Test initialization when dependencies are not available."""
        with patch.dict('sys.modules', {'faiss': None, 'sentence_transformers': None}):
            # Need to reimport to pick up the mocked modules
            import importlib
            from rag_pipeline import retriever
            importlib.reload(retriever)
            
            # When HAS_DEPS is False, should create stub mode
            if not retriever.HAS_DEPS:
                r = retriever.RAGRetriever()
                assert r.chunks == []
                assert r.index is None
                assert r.model is None

    def test_model_name_constant(self):
        """Test that MODEL_NAME is set correctly."""
        from rag_pipeline.retriever import RAGRetriever
        assert RAGRetriever.MODEL_NAME == "all-MiniLM-L6-v2"


class TestRAGRetrieverRetrieve:
    """Tests for RAGRetriever retrieve method."""

    def test_retrieve_without_index(self):
        """Test retrieve returns fallback message when index is unavailable."""
        from rag_pipeline.retriever import RAGRetriever
        
        # Create a retriever with no index
        retriever = RAGRetriever.__new__(RAGRetriever)
        retriever.index = None
        retriever.model = None
        retriever.chunks = []
        
        results = retriever.retrieve("test query", k=3)
        
        assert len(results) == 1
        assert "RAG index not available" in results[0]

    def test_retrieve_with_mock_index(self):
        """Test retrieve with a mocked FAISS index."""
        from rag_pipeline.retriever import RAGRetriever
        
        # Create a retriever instance without going through __init__
        retriever = RAGRetriever.__new__(RAGRetriever)
        retriever.chunks = [
            "Chunk about oil changes",
            "Chunk about tire maintenance",
            "Chunk about brake inspection",
        ]
        
        # Mock the model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype="float32")
        retriever.model = mock_model
        
        # Mock the FAISS index
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]]),  # scores
            np.array([[0, 1, 2]])  # indices
        )
        retriever.index = mock_index
        
        # Import faiss to get normalize_L2
        try:
            import faiss  # noqa: F401
            # Test retrieve - faiss is available so normalize_L2 will work
            results = retriever.retrieve("oil change", k=3)
            assert len(results) == 3
            assert "oil changes" in results[0]
        except ImportError:
            pytest.skip("faiss not installed")

    def test_retrieve_default_k(self):
        """Test that default k value is 3."""
        from rag_pipeline.retriever import RAGRetriever
        import inspect
        
        sig = inspect.signature(RAGRetriever.retrieve)
        k_param = sig.parameters['k']
        assert k_param.default == 3


class TestRAGRetrieverIntegration:
    """Integration tests for RAGRetriever (requires dependencies)."""

    @pytest.fixture
    def retriever_if_available(self):
        """Create a retriever if dependencies are available."""
        try:
            import faiss  # noqa: F401
            from sentence_transformers import SentenceTransformer  # noqa: F401
            from rag_pipeline.retriever import RAGRetriever
            return RAGRetriever()
        except ImportError:
            pytest.skip("faiss or sentence-transformers not installed")

    def test_retrieve_returns_list(self, retriever_if_available):
        """Test that retrieve returns a list."""
        results = retriever_if_available.retrieve("oil change", k=2)
        assert isinstance(results, list)

    def test_retrieve_respects_k(self, retriever_if_available):
        """Test that retrieve returns at most k results."""
        for k in [1, 2, 5]:
            results = retriever_if_available.retrieve("maintenance", k=k)
            assert len(results) <= k

    def test_retrieve_relevant_results(self, retriever_if_available):
        """Test that retrieve returns relevant results for a query."""
        results = retriever_if_available.retrieve("brake pads replacement", k=3)
        
        # At least one result should mention brakes
        brake_mentioned = any("brake" in r.lower() for r in results)
        assert brake_mentioned, f"Expected brake-related results, got: {results}"

    def test_retrieve_with_various_queries(self, retriever_if_available):
        """Test retrieve with different query types."""
        queries = [
            "When should I change the oil?",
            "tire pressure recommendation",
            "battery maintenance tips",
            "high mileage vehicle care",
        ]
        
        for query in queries:
            results = retriever_if_available.retrieve(query, k=2)
            assert len(results) >= 1
            assert all(isinstance(r, str) for r in results)


class TestRAGRetrieverEdgeCases:
    """Edge case tests for RAGRetriever."""

    def test_retrieve_empty_query(self):
        """Test retrieve with an empty query string."""
        from rag_pipeline.retriever import RAGRetriever
        
        retriever = RAGRetriever.__new__(RAGRetriever)
        retriever.chunks = ["Chunk 1", "Chunk 2"]
        retriever.index = None
        retriever.model = None
        
        # Should handle gracefully
        results = retriever.retrieve("", k=3)
        assert isinstance(results, list)

    def test_retrieve_large_k(self):
        """Test retrieve when k is larger than number of chunks."""
        try:
            import faiss as _faiss  # noqa: F401 - check availability
        except ImportError:
            pytest.skip("faiss not installed")
            
        from rag_pipeline.retriever import RAGRetriever
        
        retriever = RAGRetriever.__new__(RAGRetriever)
        retriever.chunks = ["Chunk 1", "Chunk 2"]
        
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]], dtype="float32")
        retriever.model = mock_model
        
        mock_index = Mock()
        # Return indices that include out-of-bounds
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7, 0.6, 0.5]]),
            np.array([[0, 1, 99, 100, 101]])  # 99, 100, 101 are out of bounds
        )
        retriever.index = mock_index
        
        # Test retrieve - faiss is available so normalize_L2 will work
        results = retriever.retrieve("test", k=5)
        # Should only return valid chunks
        assert len(results) == 2

    def test_retrieve_special_characters_in_query(self):
        """Test retrieve handles special characters in query."""
        from rag_pipeline.retriever import RAGRetriever
        
        retriever = RAGRetriever.__new__(RAGRetriever)
        retriever.chunks = ["Test chunk"]
        retriever.index = None
        retriever.model = None
        
        # Should not raise an exception
        results = retriever.retrieve("@#$%^&*()", k=1)
        assert isinstance(results, list)
