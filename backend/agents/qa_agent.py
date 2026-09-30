import requests
from backend.config import settings
from backend.rag.hybrid_retriever import HybridRetriever

class QAAgent:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.model = settings.MODEL_NAME
        self.base_url = settings.OLLAMA_BASE_URL
        
    def answer(self, question: str) -> str:
        # 1. Retrieve context
        print("Retrieving context...")
        context = self.retriever.retrieve(question)
        
        # 2. Build prompt
        prompt = f"""You are a scientific computing assistant. Get straight to the point and do not beat around the bush.
Provide direct, concise, and accurate answers with no conversational filler or preamble.
Use the following context to answer the question.
If the answer is not in the context, you can use your own knowledge but clarify that directly.

Context:
{context}

Question:
{question}

Answer:"""

        # 3. Call LLM
        print(f"Calling Ollama ({self.model})...")
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error connecting to LLM: {e}"

if __name__ == "__main__": # pragma: no cover
    import sys
    agent = QAAgent()
    print("=" * 72)
    print("📚 Hybrid RAG Scientific Q&A Agent CLI")
    print(f"Model: {agent.model} | Endpoint: {agent.base_url}")
    print("Capabilities: ChromaDB Vector Store + Kùzu Knowledge Graph")
    print("Type your scientific question below. Type 'exit', 'quit', or 'q' to end.")
    print("=" * 72)
    print()

    try:
        while True:
            try:
                user_query = input("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nExiting session. Goodbye!")
                break

            if not user_query:
                continue
            if user_query.lower() in ["exit", "quit", "q"]:
                print("Exiting session. Goodbye!")
                break

            print("\nAgent > ", end="", flush=True)
            ans = agent.answer(user_query)
            print(f"\n{ans}\n")
    except KeyboardInterrupt:
        print("\n\nSession terminated by user.")
