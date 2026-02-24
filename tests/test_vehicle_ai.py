"""
Unit tests for the VehicleAI chatbot module.
"""
import os
from unittest.mock import Mock, patch, MagicMock


class TestVehicleAIInit:
    """Tests for VehicleAI initialization."""

    def test_init_creates_memory(self):
        """Test that initialization creates a ConversationMemory instance."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {}, clear=True):
            ai = VehicleAI()
            assert ai.memory is not None
            assert len(ai.memory) == 0

    def test_init_rule_based_backend_no_keys(self):
        """Test that rule-based backend is used when no API keys are set."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {"GROQ_API_KEY": "", "HF_TOKEN": ""}, clear=True):
            ai = VehicleAI()
            assert ai.backend == "rule-based"

    def test_init_groq_backend_with_key(self):
        """Test that Groq backend is used when GROQ_API_KEY is set."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}, clear=True):
            ai = VehicleAI()
            assert ai.backend == "groq"

    def test_init_huggingface_backend_with_token(self):
        """Test that HuggingFace backend is used when HF_TOKEN is set."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {"HF_TOKEN": "test_token", "GROQ_API_KEY": ""}, clear=True):
            ai = VehicleAI()
            assert ai.backend == "huggingface"

    def test_init_with_custom_retriever(self):
        """Test initialization with a custom retriever."""
        from chatbot.vehicle_ai import VehicleAI
        
        mock_retriever = Mock()
        ai = VehicleAI(retriever=mock_retriever)
        
        assert ai._retriever == mock_retriever
        assert ai._retriever_initialized is True

    def test_system_prompt_defined(self):
        """Test that SYSTEM_PROMPT is properly defined."""
        from chatbot.vehicle_ai import VehicleAI
        
        assert hasattr(VehicleAI, "SYSTEM_PROMPT")
        assert "vehicle maintenance" in VehicleAI.SYSTEM_PROMPT.lower()


class TestVehicleAIRetrieverProperty:
    """Tests for VehicleAI retriever lazy initialization."""

    def test_retriever_lazy_init(self):
        """Test that retriever is lazily initialized on first access."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {}, clear=True):
            with patch('chatbot.vehicle_ai._rag_available', False):
                ai = VehicleAI()
                assert ai._retriever_initialized is False
                
                # Access retriever property
                _ = ai.retriever
                
                assert ai._retriever_initialized is True

    def test_retriever_reuses_existing(self):
        """Test that retriever property doesn't reinitialize if already set."""
        from chatbot.vehicle_ai import VehicleAI
        
        mock_retriever = Mock()
        ai = VehicleAI(retriever=mock_retriever)
        
        # Access multiple times
        r1 = ai.retriever
        r2 = ai.retriever
        
        assert r1 == r2 == mock_retriever


class TestVehicleAIAsk:
    """Tests for VehicleAI ask method."""

    def test_ask_adds_messages_to_memory(self):
        """Test that asking a question adds messages to memory."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {}, clear=True):
            ai = VehicleAI()
            ai._retriever = None
            ai._retriever_initialized = True
            
            initial_len = len(ai.memory)
            ai.ask("What is the oil change interval?")
            
            assert len(ai.memory) == initial_len + 2  # user + assistant

    def test_ask_returns_string(self):
        """Test that ask returns a string response."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {}, clear=True):
            ai = VehicleAI()
            ai._retriever = None
            ai._retriever_initialized = True
            
            response = ai.ask("Test question?")
            assert isinstance(response, str)
            assert len(response) > 0

    def test_ask_uses_retriever_when_available(self):
        """Test that ask uses the retriever to get context."""
        from chatbot.vehicle_ai import VehicleAI
        
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = ["Chunk 1", "Chunk 2"]
        
        with patch.dict(os.environ, {}, clear=True):
            ai = VehicleAI(retriever=mock_retriever)
            ai.ask("Test question")
            
            mock_retriever.retrieve.assert_called_once()

    def test_ask_handles_retriever_exception(self):
        """Test that ask handles retriever exceptions gracefully."""
        from chatbot.vehicle_ai import VehicleAI
        
        mock_retriever = Mock()
        mock_retriever.retrieve.side_effect = Exception("Retriever error")
        
        with patch.dict(os.environ, {}, clear=True):
            ai = VehicleAI(retriever=mock_retriever)
            
            # Should not raise, should fall back gracefully
            response = ai.ask("Test question")
            assert isinstance(response, str)


class TestVehicleAIBuildPrompt:
    """Tests for VehicleAI _build_prompt method."""

    def test_build_prompt_includes_system_prompt(self):
        """Test that prompt includes the system prompt."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {}, clear=True):
            ai = VehicleAI()
            prompt = ai._build_prompt("Test question", [])
            
            assert VehicleAI.SYSTEM_PROMPT in prompt

    def test_build_prompt_includes_context_chunks(self):
        """Test that prompt includes context chunks when provided."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {}, clear=True):
            ai = VehicleAI()
            chunks = ["Chunk 1 content", "Chunk 2 content"]
            
            prompt = ai._build_prompt("Test question", chunks)
            
            assert "Chunk 1 content" in prompt
            assert "Chunk 2 content" in prompt
            assert "Relevant Knowledge" in prompt

    def test_build_prompt_includes_question(self):
        """Test that prompt includes the user's question."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {}, clear=True):
            ai = VehicleAI()
            question = "What is the oil change interval?"
            
            prompt = ai._build_prompt(question, [])
            
            assert question in prompt

    def test_build_prompt_includes_history(self):
        """Test that prompt includes conversation history."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {}, clear=True):
            ai = VehicleAI()
            ai.memory.add("user", "Previous question")
            ai.memory.add("assistant", "Previous answer")
            
            prompt = ai._build_prompt("New question", [])
            
            assert "Previous question" in prompt
            assert "Previous answer" in prompt


class TestVehicleAIRuleBasedResponse:
    """Tests for VehicleAI _rule_based_response method."""

    def test_rule_based_with_chunks(self):
        """Test rule-based response when context chunks are available."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {}, clear=True):
            ai = VehicleAI()
            chunks = ["Oil changes every 5000 km"]
            
            response = ai._rule_based_response("oil change", chunks)
            
            assert "knowledge base" in response.lower()
            assert "5000" in response

    def test_rule_based_without_chunks(self):
        """Test rule-based response when no context chunks available."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {}, clear=True):
            ai = VehicleAI()
            
            response = ai._rule_based_response("test question", [])
            
            assert "vehicle maintenance" in response.lower()

    def test_rule_based_risk_keywords(self):
        """Test that risk-related keywords trigger specific advice."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {}, clear=True):
            ai = VehicleAI()
            chunks = ["Some context"]
            
            response = ai._rule_based_response("What is my risk score?", chunks)
            
            assert "XGBoost" in response or "risk factors" in response.lower()

    def test_rule_based_brake_keywords(self):
        """Test that brake-related keywords trigger specific advice."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {}, clear=True):
            ai = VehicleAI()
            chunks = ["Some context about brakes"]
            
            response = ai._rule_based_response("My brakes are worn", chunks)
            
            assert "safety" in response.lower() or "inspect" in response.lower()

    def test_rule_based_truncates_long_chunks(self):
        """Test that long chunks are truncated in response."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {}, clear=True):
            ai = VehicleAI()
            long_chunk = "A" * 500  # More than 300 chars
            
            response = ai._rule_based_response("test", [long_chunk])
            
            assert "..." in response


class TestVehicleAILLMBackends:
    """Tests for VehicleAI LLM API backend methods."""

    def test_call_groq_falls_back_without_httpx(self):
        """Test that Groq call falls back when httpx is unavailable."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}, clear=True):
            with patch('chatbot.vehicle_ai._httpx_available', False):
                ai = VehicleAI()
                ai._retriever = None
                ai._retriever_initialized = True
                
                response = ai._call_groq("test question", [])
                
                # Should return rule-based response
                assert isinstance(response, str)

    def test_call_huggingface_falls_back_without_httpx(self):
        """Test that HuggingFace call falls back when httpx is unavailable."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {"HF_TOKEN": "test_token", "GROQ_API_KEY": ""}, clear=True):
            with patch('chatbot.vehicle_ai._httpx_available', False):
                ai = VehicleAI()
                ai._retriever = None
                ai._retriever_initialized = True
                
                response = ai._call_huggingface("test question", [])
                
                # Should return rule-based response
                assert isinstance(response, str)

    def test_call_groq_handles_api_error(self):
        """Test that Groq handles API errors gracefully."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}, clear=True):
            with patch('chatbot.vehicle_ai._httpx_available', True):
                with patch('chatbot.vehicle_ai.httpx') as mock_httpx:
                    mock_client = MagicMock()
                    mock_client.__enter__ = Mock(return_value=mock_client)
                    mock_client.__exit__ = Mock(return_value=False)
                    mock_client.post.side_effect = Exception("API Error")
                    mock_httpx.Client.return_value = mock_client
                    
                    ai = VehicleAI()
                    ai._retriever = None
                    ai._retriever_initialized = True
                    
                    response = ai._call_groq("test question", [])
                    
                    assert "error" in response.lower() or "knowledge base" in response.lower()


class TestVehicleAIIntegration:
    """Integration tests for VehicleAI."""

    def test_full_conversation_flow(self):
        """Test a full conversation with multiple turns."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {}, clear=True):
            ai = VehicleAI()
            ai._retriever = None
            ai._retriever_initialized = True
            
            response1 = ai.ask("What is the oil change interval?")
            assert isinstance(response1, str)
            assert len(ai.memory) == 2
            
            response2 = ai.ask("What about synthetic oil?")
            assert isinstance(response2, str)
            assert len(ai.memory) == 4

    def test_memory_persistence_across_questions(self):
        """Test that memory persists across multiple questions."""
        from chatbot.vehicle_ai import VehicleAI
        
        with patch.dict(os.environ, {}, clear=True):
            ai = VehicleAI()
            ai._retriever = None
            ai._retriever_initialized = True
            
            ai.ask("Question 1")
            ai.ask("Question 2")
            ai.ask("Question 3")
            
            context = ai.memory.get_context_string()
            assert "Question 1" in context
            assert "Question 2" in context
            assert "Question 3" in context
