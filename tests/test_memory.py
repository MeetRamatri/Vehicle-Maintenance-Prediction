"""
Unit tests for the ConversationMemory module.
"""
import pytest
from chatbot.memory import ConversationMemory


class TestConversationMemory:
    """Tests for ConversationMemory class."""

    def test_init_default_max_turns(self):
        """Test default initialization with max_turns=10."""
        memory = ConversationMemory()
        assert memory.max_turns == 10
        assert len(memory.history) == 0

    def test_init_custom_max_turns(self):
        """Test initialization with custom max_turns."""
        memory = ConversationMemory(max_turns=5)
        assert memory.max_turns == 5

    def test_add_user_message(self):
        """Test adding a user message."""
        memory = ConversationMemory()
        memory.add("user", "Hello")
        
        assert len(memory) == 1
        assert memory.history[0]["role"] == "user"
        assert memory.history[0]["content"] == "Hello"

    def test_add_assistant_message(self):
        """Test adding an assistant message."""
        memory = ConversationMemory()
        memory.add("assistant", "Hi there!")
        
        assert len(memory) == 1
        assert memory.history[0]["role"] == "assistant"
        assert memory.history[0]["content"] == "Hi there!"

    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        memory = ConversationMemory()
        memory.add("user", "Question 1")
        memory.add("assistant", "Answer 1")
        memory.add("user", "Question 2")
        memory.add("assistant", "Answer 2")
        
        assert len(memory) == 4

    def test_history_trimming(self):
        """Test that history is trimmed when exceeding max_turns * 2."""
        memory = ConversationMemory(max_turns=2)  # max 4 messages
        
        # Add 6 messages (3 turns)
        for i in range(3):
            memory.add("user", f"Question {i}")
            memory.add("assistant", f"Answer {i}")
        
        # Should be trimmed to last 4 messages (2 turns)
        assert len(memory) == 4
        assert memory.history[0]["content"] == "Question 1"
        assert memory.history[-1]["content"] == "Answer 2"

    def test_get_context_string_empty(self):
        """Test get_context_string with empty history."""
        memory = ConversationMemory()
        assert memory.get_context_string() == ""

    def test_get_context_string_with_messages(self):
        """Test get_context_string with messages."""
        memory = ConversationMemory()
        memory.add("user", "What is oil change interval?")
        memory.add("assistant", "Every 5,000 to 7,500 km.")
        
        context = memory.get_context_string()
        assert "User: What is oil change interval?" in context
        assert "Assistant: Every 5,000 to 7,500 km." in context

    def test_get_context_string_format(self):
        """Test that context string uses proper formatting."""
        memory = ConversationMemory()
        memory.add("user", "Hello")
        memory.add("assistant", "Hi")
        
        context = memory.get_context_string()
        lines = context.split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("User:")
        assert lines[1].startswith("Assistant:")

    def test_clear(self):
        """Test clearing the conversation history."""
        memory = ConversationMemory()
        memory.add("user", "Test message")
        memory.add("assistant", "Test response")
        
        assert len(memory) == 2
        memory.clear()
        assert len(memory) == 0
        assert memory.history == []

    def test_len(self):
        """Test __len__ method."""
        memory = ConversationMemory()
        assert len(memory) == 0
        
        memory.add("user", "Test")
        assert len(memory) == 1
        
        memory.add("assistant", "Response")
        assert len(memory) == 2

    def test_with_sample_history(self, sample_conversation_history):
        """Test memory with sample conversation history fixture."""
        memory = ConversationMemory()
        for item in sample_conversation_history:
            memory.add(item["role"], item["content"])
        
        assert len(memory) == 4
        context = memory.get_context_string()
        assert "oil change interval" in context.lower()
        assert "synthetic oil" in context.lower()

    def test_boundary_max_turns_one(self):
        """Test with max_turns=1 (edge case)."""
        memory = ConversationMemory(max_turns=1)
        
        memory.add("user", "Q1")
        memory.add("assistant", "A1")
        memory.add("user", "Q2")
        memory.add("assistant", "A2")
        
        # Should keep only last 2 messages
        assert len(memory) == 2
        assert memory.history[0]["content"] == "Q2"
        assert memory.history[1]["content"] == "A2"

    def test_large_messages(self):
        """Test handling of large message content."""
        memory = ConversationMemory()
        large_content = "A" * 10000  # 10KB message
        
        memory.add("user", large_content)
        assert memory.history[0]["content"] == large_content

    def test_special_characters_in_messages(self):
        """Test handling of special characters."""
        memory = ConversationMemory()
        special_content = "Test with special chars: @#$%^&*()_+=<>?/\\|{}[]"
        
        memory.add("user", special_content)
        assert memory.history[0]["content"] == special_content

    def test_unicode_messages(self):
        """Test handling of unicode characters."""
        memory = ConversationMemory()
        unicode_content = "Testing unicode: 你好 مرحبا 🚗🔧"
        
        memory.add("user", unicode_content)
        assert memory.history[0]["content"] == unicode_content
