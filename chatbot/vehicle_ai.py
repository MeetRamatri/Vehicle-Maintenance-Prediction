"""
Vehicle AI Chatbot Module
Conversational AI agent that combines RAG retrieval with LLM reasoning
for vehicle maintenance questions. Supports Groq and HuggingFace backends,
with a smart rule-based fallback when no API key is configured.
"""
import os
import json
from typing import Optional

from chatbot.memory import ConversationMemory

# Try importing RAG retriever
try:
    from rag_pipeline.retriever import RAGRetriever
    _rag_available = True
except Exception:
    _rag_available = False

# Try importing HTTP client for LLM API calls
try:
    import httpx
    _httpx_available = True
except ImportError:
    _httpx_available = False


class VehicleAI:
    """
    Phase 10: Conversational AI
    - Memory Handling via ConversationMemory
    - RAG-augmented contextual reasoning
    - LLM backend (Groq / HuggingFace) with rule-based fallback
    """

    SYSTEM_PROMPT = (
        "You are an expert vehicle maintenance assistant. You help fleet managers "
        "and vehicle owners understand maintenance needs, interpret risk scores, "
        "and provide actionable service recommendations. "
        "Use the provided context to give accurate, specific answers. "
        "If you don't know something, say so rather than guessing. "
        "Keep responses concise but informative."
    )

    def __init__(self):
        self.memory = ConversationMemory(max_turns=10)

        # Initialize RAG retriever
        if _rag_available:
            try:
                self.retriever = RAGRetriever()
            except Exception as e:
                print(f"Warning: RAG retriever init failed: {e}")
                self.retriever = None
        else:
            self.retriever = None

        # Determine LLM backend
        self.groq_key = os.environ.get("GROQ_API_KEY", "")
        self.hf_token = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_TOKEN", ""))

        if self.groq_key:
            self.backend = "groq"
            print("VehicleAI: Using Groq API backend")
        elif self.hf_token:
            self.backend = "huggingface"
            print("VehicleAI: Using HuggingFace Inference API backend")
        else:
            self.backend = "rule-based"
            print("VehicleAI: No API keys found. Using rule-based fallback. "
                  "Set GROQ_API_KEY or HF_TOKEN for LLM-powered responses.")

    def ask(self, question: str) -> str:
        """Process a user question and return a response."""
        # Retrieve relevant context
        context_chunks = []
        if self.retriever:
            try:
                context_chunks = self.retriever.retrieve(question, k=3)
            except Exception:
                pass

        # Add question to memory
        self.memory.add("user", question)

        # Generate response
        if self.backend == "groq":
            response = self._call_groq(question, context_chunks)
        elif self.backend == "huggingface":
            response = self._call_huggingface(question, context_chunks)
        else:
            response = self._rule_based_response(question, context_chunks)

        # Add response to memory
        self.memory.add("assistant", response)
        return response

    def _build_prompt(self, question: str, context_chunks: list[str]) -> str:
        """Build a full prompt with system context, RAG chunks, and history."""
        parts = [self.SYSTEM_PROMPT]

        if context_chunks:
            parts.append("\n--- Relevant Knowledge ---")
            for i, chunk in enumerate(context_chunks, 1):
                parts.append(f"{i}. {chunk}")
            parts.append("--- End Knowledge ---\n")

        history = self.memory.get_context_string()
        if history:
            parts.append(f"Conversation so far:\n{history}")

        parts.append(f"\nUser question: {question}")
        parts.append("\nProvide a helpful, concise answer:")
        return "\n".join(parts)

    def _call_groq(self, question: str, context_chunks: list[str]) -> str:
        """Call Groq API with Llama 3."""
        if not _httpx_available:
            return self._rule_based_response(question, context_chunks)

        prompt = self._build_prompt(question, context_chunks)
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.groq_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "llama3-8b-8192",
                        "messages": [
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 512,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"LLM API error: {e}. Falling back to knowledge base.\n\n" + self._rule_based_response(question, context_chunks)

    def _call_huggingface(self, question: str, context_chunks: list[str]) -> str:
        """Call HuggingFace Inference API."""
        if not _httpx_available:
            return self._rule_based_response(question, context_chunks)

        prompt = self._build_prompt(question, context_chunks)
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct",
                    headers={
                        "Authorization": f"Bearer {self.hf_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "inputs": prompt,
                        "parameters": {"max_new_tokens": 512, "temperature": 0.3},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get("generated_text", "").strip()
                return str(data)
        except Exception as e:
            return f"LLM API error: {e}. Falling back to knowledge base.\n\n" + self._rule_based_response(question, context_chunks)

    def _rule_based_response(self, question: str, context_chunks: list[str]) -> str:
        """Smart rule-based fallback using RAG chunks and keyword matching."""
        q_lower = question.lower()

        # Build response from relevant RAG chunks
        if context_chunks:
            response_parts = ["Based on our vehicle maintenance knowledge base:\n"]
            for i, chunk in enumerate(context_chunks, 1):
                # Truncate long chunks for readability
                display = chunk[:300] + "..." if len(chunk) > 300 else chunk
                response_parts.append(f"  {i}. {display}")

            # Add targeted advice based on keywords
            if any(word in q_lower for word in ["risk", "score", "predict"]):
                response_parts.append(
                    "\nOur ML model uses XGBoost and identifies these top risk factors: "
                    "Reported Issues, Brake Condition (Worn Out), Battery Status (Weak), "
                    "Service History, and Maintenance History."
                )
            elif any(word in q_lower for word in ["oil", "change", "service"]):
                response_parts.append(
                    "\nTip: Regular servicing based on manufacturer schedules "
                    "is the best way to prevent costly repairs."
                )
            elif any(word in q_lower for word in ["brake", "tire", "battery"]):
                response_parts.append(
                    "\nThese components are critical safety items and top predictors "
                    "of maintenance needs. Inspect them regularly."
                )

            return "\n".join(response_parts)
        else:
            return (
                "I'm your vehicle maintenance AI assistant. I can help with questions about "
                "maintenance schedules, risk factors, service recommendations, and interpreting "
                "vehicle health data. Please ask a specific question about vehicle maintenance."
            )


if __name__ == "__main__":
    ai = VehicleAI()
    questions = [
        "When should I change the oil?",
        "What are the top risk factors for maintenance?",
        "My brakes are worn out. What should I do?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {ai.ask(q)}")
        print("-" * 60)
