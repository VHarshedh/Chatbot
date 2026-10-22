import os
import re
import json
import sqlite3
import hashlib
from typing import List, Dict, Any, Optional
import chromadb
from backend.config import settings
from backend.rag.embeddings import OllamaEmbeddings

class ChatMemoryStore:
    """
    Episodic Cross-Session RAG Memory:
    Extracts, indexes, and semantically retrieves conversational exchanges across all chat sessions
    using an embedded ChromaDB collection ('chat_history_memory').
    """
    _instance = None

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embeddings = OllamaEmbeddings()
        self.collection = self.client.get_or_create_collection(name="chat_history_memory")

    @classmethod
    def get_instance(cls, persist_dir: Optional[str] = None):
        if cls._instance is None:
            cls._instance = cls(persist_dir)
        return cls._instance

    @staticmethod
    def clean_text(text: str) -> str:
        """Removes collapsible thinking HTML traces and excessively long logs"""
        if not text:
            return ""
        # Remove thinking details
        cleaned = re.sub(r'<details class="thinking-details">.*?</details>', '', text, flags=re.DOTALL).strip()
        # Clean excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        return cleaned

    def index_turn(
        self,
        session_id: int,
        user_message: str,
        assistant_message: str,
        agent_used: str = "Assistant",
        session_title: Optional[str] = None,
        turn_id: Optional[str] = None,
        execution_required: Optional[bool] = None,
        exit_code: Optional[int] = None,
        evaluation_score: Optional[float] = None
    ) -> bool:
        """
        Indexes a single user-assistant exchange into ChromaDB cross-session memory.
        """
        if not user_message or not assistant_message:
            return False

        if execution_required is None:
            # Unknown historical code is never assumed to have executed successfully.
            execution_required = bool(re.search(r"```|\b(run|execute)\b", assistant_message + " " + user_message, re.I))
        verified = not execution_required or exit_code == 0 or (evaluation_score is not None and evaluation_score >= 80)
        if not verified:
            return False

        clean_user = user_message.strip()
        clean_asst = self.clean_text(assistant_message)
        if not clean_asst:
            clean_asst = assistant_message.strip()

        # Format memory document for semantic indexing
        doc_text = (
            f"Session #{session_id} ({session_title or 'Chat Session'})\n"
            f"User Question: {clean_user}\n"
            f"Assistant Response: {clean_asst}"
        )

        turn_hash = hashlib.sha256(f"{session_id}:{clean_user}".encode("utf-8")).hexdigest()[:12]
        doc_id = turn_id or f"turn_s{session_id}_{turn_hash}"
        metadata = {
            "session_id": int(session_id),
            "session_title": str(session_title or f"Session {session_id}")[:120],
            "agent_used": str(agent_used or "Assistant"),
            "user_snippet": clean_user[:400],
            "assistant_snippet": clean_asst[:600],
            "verified_execution": True
        }

        try:
            emb = self.embeddings.embed_documents([doc_text])
            self.collection.upsert(
                documents=[doc_text],
                embeddings=emb,
                metadatas=[metadata],
                ids=[doc_id]
            )
            return True
        except Exception as e:
            print(f"Notice: Failed to index chat turn into memory: {e}")
            return False

    def ingest_all_sessions(self, sqlite_path: Optional[str] = None) -> int:
        """
        Scans all sessions and messages in SQLite, pairs conversational exchanges,
        and batch-ingests them into the ChromaDB chat_history_memory collection.
        Returns the count of conversational turns ingested.
        """
        db_path = sqlite_path or settings.SQLITE_PATH
        if not os.path.exists(db_path):
            print(f"Notice: SQLite database not found at {db_path}")
            return 0

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query all sessions
        cursor.execute("SELECT id, title FROM sessions ORDER BY id ASC")
        sessions = [dict(r) for r in cursor.fetchall()]
        if not sessions:
            conn.close()
            return 0

        count = 0
        try:
            for session in sessions:
                cursor.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY id", (session["id"],))
                messages = [dict(row) for row in cursor.fetchall()]
                for user, assistant in zip(messages, messages[1:]):
                    if user["role"] != "user" or assistant["role"] != "assistant":
                        continue
                    try:
                        quality = json.loads(assistant.get("tool_calls_json") or "{}")
                    except (ValueError, TypeError):
                        quality = {}
                    # Legacy traces are not reliable execution evidence.
                    if isinstance(quality, list):
                        quality = next((item for item in reversed(quality) if isinstance(item, dict) and "memory_quality" in item), {})
                    quality = quality.get("memory_quality", {}) if isinstance(quality, dict) else {}
                    count += self.index_turn(
                        session_id=session["id"], user_message=user["content"],
                        assistant_message=assistant["content"],
                        agent_used=assistant.get("agent_used"), session_title=session["title"],
                        execution_required=quality.get("execution_required"),
                        exit_code=quality.get("exit_code"),
                        evaluation_score=quality.get("evaluation_score"),
                    )
            return count
        finally:
            conn.close()

    def search_memory(
        self,
        query: str,
        n_results: int = 3,
        current_session_id: Optional[int] = None,
        max_distance: float = 1.15
    ) -> List[Dict[str, Any]]:
        """
        Retrieves relevant past conversational exchanges from other sessions.
        Filters out turns from current_session_id to prevent echoing immediate context.
        """
        if not query or not query.strip():
            return []

        try:
            count = self.collection.count()
            if count == 0:
                return []

            query_emb = self.embeddings.embed_query(query)
            n_fetch = min(n_results * 3, count)
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=n_fetch,
                where={"verified_execution": True}
            )

            matches = []
            if results and results.get("documents") and len(results["documents"]) > 0:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

                for doc, meta, dist in zip(docs, metas, dists):
                    # Filter out current session turns (which are already in local sliding window)
                    sid = meta.get("session_id")
                    if current_session_id is not None and sid == current_session_id:
                        continue
                    if dist > max_distance:
                        continue

                    matches.append({
                        "session_id": sid,
                        "session_title": meta.get("session_title", "Past Session"),
                        "agent_used": meta.get("agent_used", "Assistant"),
                        "user_snippet": meta.get("user_snippet", ""),
                        "assistant_snippet": meta.get("assistant_snippet", ""),
                        "distance": dist,
                        "content": doc
                    })
                    if len(matches) >= n_results:
                        break

            return matches
        except Exception as e:
            print(f"Notice: Chat memory search failed: {e}")
            return []

    def format_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        """Formats retrieved cross-session memories for prompt context injection"""
        if not memories:
            return ""
        lines = ["--- RELEVANT PAST CHAT SESSIONS MEMORY ---"]
        for m in memories:
            title = m.get("session_title", "Past Chat")
            sid = m.get("session_id", "?")
            user_part = m.get("user_snippet", "")
            asst_part = m.get("assistant_snippet", "")
            lines.append(f"[Session #{sid}: {title}]")
            lines.append(f"User: {user_part}")
            lines.append(f"Assistant: {asst_part}\n")
        return "\n".join(lines).strip()

def get_chat_memory_store() -> ChatMemoryStore:
    return ChatMemoryStore.get_instance()

if __name__ == "__main__":
    store = ChatMemoryStore()
    count = store.ingest_all_sessions()
    print(f"Total sessions ingested: {count}")
