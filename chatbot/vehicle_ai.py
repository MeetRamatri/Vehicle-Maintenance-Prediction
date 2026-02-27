class VehicleAI:
    """
    Phase 10: Conversational AI
    - Memory Handling
    - Contextual Reasoning
    """
    def __init__(self):
        self.memory = []
        
    def ask(self, question):
        self.memory.append(question)
        return f"AI: Based on your vehicle history, {question} is worth investigating."

if __name__ == "__main__":
    ai = VehicleAI()
    print(ai.ask("Explain the risk score."))
