import os
import re
import json
import sqlite3
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
        turn_id: Optional[str] = None
    ) -> bool:
        """
        Indexes a single user-assistant exchange into ChromaDB cross-session memory.
        """
        if not user_message or not assistant_message:
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

        doc_id = turn_id or f"turn_s{session_id}_{abs(hash(clean_user[:80]))}"
        metadata = {
            "session_id": int(session_id),
            "session_title": str(session_title or f"Session {session_id}")[:120],
            "agent_used": str(agent_used or "Assistant"),
            "user_snippet": clean_user[:400],
            "assistant_snippet": clean_asst[:600]
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

        all_docs = []
        all_metas = []
        all_ids = []

        for s in sessions:
            sid = s["id"]
            stitle = s["title"] or f"Session {sid}"

            cursor.execute(
                "SELECT id, role, content, agent_used, timestamp FROM messages WHERE session_id = ? ORDER BY id ASC",
                (sid,)
            )
            msgs = [dict(r) for r in cursor.fetchall()]

            # Group messages into conversational exchanges (User -> Assistant)
            i = 0
            turn_idx = 1
            while i < len(msgs):
                if msgs[i]["role"] == "user":
                    user_text = msgs[i]["content"] or ""
                    asst_text = ""
                    agent_used = "Assistant"
                    if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
                        asst_text = msgs[i + 1]["content"] or ""
                        agent_used = msgs[i + 1].get("agent_used") or "Assistant"
                        i += 2
                    else:
                        i += 1

                    if user_text and asst_text:
                        clean_asst = self.clean_text(asst_text)
                        doc_text = (
                            f"Session #{sid} ({stitle})\n"
                            f"User Question: {user_text.strip()}\n"
                            f"Assistant Response: {clean_asst}"
                        )
                        doc_id = f"turn_s{sid}_m{turn_idx}"
                        metadata = {
                            "session_id": int(sid),
                            "session_title": str(stitle)[:120],
                            "agent_used": str(agent_used),
                            "user_snippet": user_text.strip()[:400],
                            "assistant_snippet": clean_asst[:600]
                        }
                        all_docs.append(doc_text)
                        all_metas.append(metadata)
                        all_ids.append(doc_id)
                        turn_idx += 1
                else:
                    i += 1

        conn.close()

        if not all_docs:
            return 0

        try:
            print(f"Embedding and ingesting {len(all_docs)} past conversational turns into ChromaDB memory...")
            embeddings = self.embeddings.embed_documents(all_docs)
            self.collection.upsert(
                documents=[all_docs[i] for i in range(len(all_docs))],
                embeddings=embeddings,
                metadatas=[all_metas[i] for i in range(len(all_metas))],
                ids=[all_ids[i] for i in range(len(all_ids))]
            )
            print(f"Successfully ingested {len(all_docs)} past conversational turns into memory.")
            return len(all_docs)
        except Exception as e:
            print(f"Error ingesting past sessions into memory: {e}")
            return 0

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
                n_results=n_fetch
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
