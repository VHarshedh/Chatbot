import re
import requests
from backend.config import settings
from backend.rag.hybrid_retriever import HybridRetriever
from backend.tools.web_tools import search_web, read_url_content

QA_SYSTEM_PROMPT = (
    "SYSTEM DIRECTIVE: You are SciCompute AI, an authoritative scientific computing and computational materials assistant "
    "specialized in Density Functional Theory (DFT), Molecular Dynamics (LAMMPS), Machine Learning Potentials (DeepMD-kit), and ASE.\n"
    "\n"
    "CORE PROTOCOLS:\n"
    "1. Grounding & Provenance: Ground your answer strictly on the provided Hybrid RAG context (ChromaDB vector documents and Kùzu knowledge graph relationships). "
    "Prioritize retrieved parameters, formulas, cutoffs, and data structures.\n"
    "2. Context Boundaries: If the context contains the answer, synthesize it directly with high technical precision. "
    "If the question exceeds the context, leverage your computational materials science knowledge but state this boundary concisely.\n"
    "3. Mathematical Precision: Formulate all physical equations and parameters using standard LaTeX ($...$ for inline, $$...$$ for block display).\n"
    "4. Concise Execution: Get straight to the point and do not beat around the bush. Provide zero introductory pleasantries, filler phrases, or conversational preambles. "
    "Deliver the direct answer followed by rigorous technical elaboration."
)

class QAAgent:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.model = settings.MODEL_NAME
        self.base_url = settings.OLLAMA_BASE_URL

    def search_web(self, query: str, max_results: int = 3) -> list:
        """Searches the web via DuckDuckGo."""
        return search_web(query, max_results=max_results)

    def read_url_content(self, url: str) -> str:
        """Fetches and extracts text from a specified URL."""
        return read_url_content(url)
        
    def answer_question(self, question: str) -> str:
        """Alias for answer() method."""
        return self.answer(question)

    def answer(self, question: str) -> str:
        # 1. Retrieve context
        print("Retrieving context...")
        context = self.retriever.retrieve(question)

        # 2. Enrich with web tools if requested or URLs present
        web_context = []
        if re.search(r"\b(search (?:the )?web|web search|look up online|search online)\b", question, re.I):
            clean_q = re.sub(r"(?i)\b(search (?:the )?web (?:for)?|look up online (?:for)?|web search)\b", "", question).strip() or question
            web_hits = self.search_web(clean_q, max_results=3)
            if web_hits and not web_hits[0].get("error"):
                formatted = "\n".join(f"- {h.get('title')}: {h.get('body')} ({h.get('href')})" for h in web_hits if h.get("title"))
                if formatted:
                    web_context.append(f"Web Search Results:\n{formatted}")

        urls = re.findall(r"https?://[^\s)\]]+", question)
        for u in urls[:2]:
            u_text = self.read_url_content(u)
            if u_text and not u_text.startswith("Error"):
                web_context.append(f"Content from {u}:\n{u_text[:2000]}")

        if web_context:
            context = f"{context}\n\n" + "\n\n".join(web_context)
        
        # 2. Build user prompt
        user_prompt = f"""Retrieved Domain Context:
{context}

Question:
{question}

Provide an authoritative, direct scientific answer:"""

        # 3. Call LLM with native system role and options
        print(f"Calling Ollama ({self.model})...")
        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "keep_alive": settings.OLLAMA_KEEP_ALIVE,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_ctx": settings.NUM_CTX,
                        "temperature": 0.2,
                        "num_gpu": settings.NUM_GPU,
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            if "message" in data:
                return data["message"].get("content", "")
            return data.get("response", "")
        except Exception as e:
            # Fallback to /api/generate for backward compatibility or unit test mocks
            try:
                gen_payload = {
                    "model": self.model,
                    "keep_alive": settings.OLLAMA_KEEP_ALIVE,
                    "system": QA_SYSTEM_PROMPT,
                    "prompt": f"{QA_SYSTEM_PROMPT}\n\n{user_prompt}",
                    "stream": False
                }
                fallback_res = requests.post(
                    f"{self.base_url}/api/generate",
                    json=gen_payload,
                    timeout=120
                )
                fallback_res.raise_for_status()
                fdata = fallback_res.json()
                return fdata.get("response") or fdata.get("message", {}).get("content", "")
            except Exception:
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
