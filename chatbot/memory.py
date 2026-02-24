"""
Conversation Memory Module
Maintains chat history with a sliding window to provide context for the AI chatbot.
"""


class ConversationMemory:
    """Manages conversation history with a configurable window size."""

    def __init__(self, max_turns: int = 10):
        self.history: list[dict[str, str]] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        """Add a message to history. role is 'user' or 'assistant'."""
        self.history.append({"role": role, "content": content})
        # Trim to keep only the last max_turns exchanges
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_context_string(self) -> str:
        """Return conversation history as a formatted string for prompt context."""
        if not self.history:
            return ""
        lines = []
        for msg in self.history:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)

    def clear(self):
        """Clear all conversation history."""
        self.history = []

    def __len__(self):
        return len(self.history)
