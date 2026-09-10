import os
import json
from pathlib import Path
from typing import List, Dict, Any
from backend.config import settings
from backend.rag.vectorstore import VectorStore
from backend.rag.knowledge_graph import KnowledgeGraph

def extract_notebook_cells(notebook_path: str) -> List[Dict[str, Any]]:
    """
    Parses a Jupyter Notebook (.ipynb) file and extracts clean code and markdown cells.
    Completely filters out outputs, base64 images, and noisy execution logs.
    """
    path = Path(notebook_path)
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)
    except Exception as e:
        print(f"Error reading notebook {notebook_path}: {e}")
        return []

    cells = nb_data.get("cells", [])
    extracted_docs = []
    nb_name = path.name

    # Determine scientific domain from folder structure
    domain = "General"
    parent_name = path.parent.name
    if "DFT" in parent_name or "DFT" in str(path):
        domain = "DFT"
    elif "Molecular Dynamics" in parent_name or "MD" in str(path):
        domain = "Molecular Dynamics"
    elif "Training" in parent_name:
        domain = "DeepMD Training"
    elif "Analysis" in parent_name:
        domain = "Interfacial Analysis"

    for idx, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "unknown")
        source = cell.get("source", [])
        if isinstance(source, list):
            source_text = "".join(source).strip()
        else:
            source_text = str(source).strip()

        # Skip empty cells
        if not source_text:
            continue

        header = f"### [Notebook: {nb_name} | Domain: {domain} | Cell {idx} ({cell_type.upper()})]\n"
        full_content = f"{header}{source_text}"

        extracted_docs.append({
            "content": full_content,
            "metadata": {
                "source": str(path),
                "notebook_name": nb_name,
                "domain": domain,
                "cell_index": idx,
                "cell_type": cell_type,
                "doc_type": "notebook_cell"
            },
            "id": f"nb_{path.stem}_cell_{idx}"
        })

    return extracted_docs

def scan_workspace_notebooks(workspace_root: str = None) -> List[str]:
    """Finds all valid .ipynb files in the workspace, excluding checkpoints and virtual environments."""
    root = Path(workspace_root or settings.WORKSPACE_ROOT)
    notebook_paths = []
    
    if not root.exists():
        return []

    for path in root.rglob("*.ipynb"):
        # Ignore checkpoints and cache folders
        if ".ipynb_checkpoints" in path.parts or "venv" in path.parts or ".venv" in path.parts:
            continue
        notebook_paths.append(str(path))

    return notebook_paths

def ingest_notebooks_to_vectorstore(notebook_paths: List[str] = None) -> int:
    """Ingests all cleaned notebook cells into the ChromaDB VectorStore."""
    if notebook_paths is None:
        notebook_paths = scan_workspace_notebooks()

    all_docs = []
    for nb_path in notebook_paths:
        cells = extract_notebook_cells(nb_path)
        all_docs.extend(cells)

    if not all_docs:
        print("No notebook cells found to ingest.")
        return 0

    texts = [doc["content"] for doc in all_docs]
    metadatas = [doc["metadata"] for doc in all_docs]
    ids = [doc["id"] for doc in all_docs]

    print(f"Ingesting {len(all_docs)} notebook cells across {len(notebook_paths)} notebooks into VectorStore...")
    store = VectorStore()
    store.add_documents(texts, metadatas, ids)
    print("Notebook vector ingestion complete.")
    return len(all_docs)

def register_notebooks_in_knowledge_graph(notebook_paths: List[str] = None):
    """
    Registers notebook nodes and domain associations in the Kùzu Knowledge Graph.
    """
    if notebook_paths is None:
        notebook_paths = scan_workspace_notebooks()

    kg = KnowledgeGraph()

    # Ensure Notebook schema exists in Kùzu
    try:
        kg.conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Notebook (name STRING, path STRING, domain STRING, PRIMARY KEY (name))"
        )
        kg.conn.execute(
            "CREATE REL TABLE IF NOT EXISTS operates_on (FROM Notebook TO Material)"
        )
        kg.conn.execute(
            "CREATE REL TABLE IF NOT EXISTS targets_system (FROM Notebook TO System)"
        )
    except Exception as e:
        if "already exists" not in str(e).lower():
            print(f"Warning creating notebook graph schema: {e}")

    # Register each notebook and connect to relevant materials / systems
    for nb_path in notebook_paths:
        p = Path(nb_path)
        name = p.name
        safe_path = str(p).replace("\\", "/")

        domain = "General"
        if "DFT" in str(p):
            domain = "DFT"
        elif "Molecular Dynamics" in str(p):
            domain = "Molecular Dynamics"
        elif "Training" in str(p):
            domain = "DeepMD Training"
        elif "Analysis" in str(p):
            domain = "Interfacial Analysis"

        try:
            kg.conn.execute(
                f"MERGE (n:Notebook {{name: '{name}', path: '{safe_path}', domain: '{domain}'}})"
            )

            # Link DFT notebook to Pt Material
            if domain == "DFT":
                kg.conn.execute(
                    f"MATCH (n:Notebook {{name: '{name}'}}), (m:Material {{symbol: 'Pt'}}) "
                    f"MERGE (n)-[:operates_on]->(m)"
                )
            # Link MD and Training notebooks to Water System
            elif domain in ("Molecular Dynamics", "DeepMD Training"):
                kg.conn.execute(
                    f"MATCH (n:Notebook {{name: '{name}'}}), (s:System {{name: 'Water'}}) "
                    f"MERGE (n)-[:targets_system]->(s)"
                )
        except Exception as e:
            print(f"Error registering notebook {name} in Knowledge Graph: {e}")

    print("Notebook Knowledge Graph registration complete.")

def ingest_all_notebook_data():
    """Runs dual ingestion: clean code/text cells into VectorStore and structural relationships into Knowledge Graph."""
    notebooks = scan_workspace_notebooks()
    print(f"Found {len(notebooks)} notebooks in workspace: {[Path(n).name for n in notebooks]}")
    
    # 1. Vector Store Ingestion (Code & Explanations)
    ingest_notebooks_to_vectorstore(notebooks)

    # 2. Knowledge Graph Registration (Entities & Relationships)
    register_notebooks_in_knowledge_graph(notebooks)

if __name__ == "__main__":
    ingest_all_notebook_data()
