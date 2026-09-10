import os
import chromadb
from typing import List, Dict, Any
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.config import settings
from backend.rag.embeddings import OllamaEmbeddings

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.embeddings = OllamaEmbeddings()
        self.collection = self.client.get_or_create_collection(name="chatbox_docs")
        
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        # Embed and add
        embeddings = self.embeddings.embed_documents(texts)
        self.collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
    def semantic_search(self, query: str, n_results: int = 5, filter: dict = None) -> List[Dict]:
        query_embedding = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results and results['distances'] else 0.0
                })
        return formatted_results

def ingest_readmes():
    """Ingest local READMEs and manuals into ChromaDB"""
    store = VectorStore()
    
    # Load all markdown and python files from the workspace with UTF-8 encoding
    loader = DirectoryLoader(
        settings.WORKSPACE_ROOT,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True
    )
    py_loader = DirectoryLoader(
        settings.WORKSPACE_ROOT,
        glob="**/*.py",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True
    )
    try:
        docs = loader.load()
        docs.extend(py_loader.load())
    except Exception as e:
        print(f"Error loading documents: {e}")
        return
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    texts = [split.page_content for split in splits]
    metadatas = [split.metadata for split in splits]
    ids = [f"doc_{i}" for i in range(len(splits))]
    
    if texts:
        print(f"Ingesting {len(texts)} chunks...")
        store.add_documents(texts, metadatas, ids)
        print("Ingestion complete.")
    else:
        print("No documents found.")

if __name__ == "__main__":
    ingest_readmes()
