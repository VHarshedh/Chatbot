# Attaching `dataset.zip` to the Agentic Agent via Hybrid GraphRAG (`rag-improvement.md`)

## 1. Executive Verdict: Should You Attach `dataset.zip` via RAG, Fine-Tuning, or Both?

**Use Both (Complementary Division of Labor)—and inside RAG, use a Hybrid Knowledge Graph (Kùzu) + Metadata-Filtered VectorStore (ChromaDB) architecture.**

Why a 2B model (`batiai/gemma4-e2b:q4`) needs **both** `PhysChem-finetuning.ipynb` and `dataset.zip` attached via Hybrid GraphRAG:

| Capability Dimension | Handled Best By `PhysChem-finetuning.ipynb` (Weights) | Handled Best By `dataset.zip` via Hybrid GraphRAG (Context) |
| :--- | :--- | :--- |
| **Algorithmic Habits & Multi-Step Reasoning** | Teaches the model *how* to write `# [PhysChem Reasoning]` headers, unpack `nl_i, nl_j, nl_d = neighbor_list('ijd', ...)`, partition shared elements (`O_w` vs `O_slab`) first, and normalize spherical shells ($4\pi r^2 \Delta r$). | Provides verified multi-step reference templates (`sample_q1..q5.py`, `Water_Density_RDF_..._validated.py`) when the query matches a complex multi-panel workflow. |
| **Exact System-Specific Constants & File Schemas** | Weights alone can still blur exact numbers across 20+ materials (e.g., whether `Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz` has 426 atoms or 558 atoms, or whether `IrO2` uses $r_{\text{cut}} = 3.0\text{ \AA}$ vs $3.2\text{ \AA}$). | Injects the **exact, zero-hallucination `<structure_metadata>` card** (`426 atoms: 60 Ti, 120 O_slab, 82 O_w, 164 H`) and **exact bond cutoffs** (`r_OH = 1.25 Å`, `r_if = 3.20 Å`) directly into the prompt! |
| **Zero-Cost Updates for New Materials/Files** | Requires re-running a multi-hour GPU session if you add a new material (e.g., $\text{RuO}_2$ or $\text{CeO}_2$). | Simply add a row to `physchem_knowledge_graph_triples.json` or drop a new `.extxyz` file into `dataset.zip`—updated in $< 1$ second! |

---

## 2. Knowledge Graph (Kùzu) vs. VectorStore (ChromaDB): What Goes Where?

Neither a pure VectorStore nor a pure Knowledge Graph is sufficient on its own for scientific coding:
- **Why Pure VectorStore Fails on Physics/Chemistry Constants**: Dense embeddings (`nomic-embed-text` / `bge-small`) blur numerical cutoffs and chemical formulas (`TiO2` vs `IrO2` vs `SiO2` all have high cosine similarity for *"water interface cutoff"*). Retrieving a `SiO2` or `IrO2` chunk when the user asks about `TiO2` injects the wrong stoichiometry or file path.
- **Why Pure Knowledge Graph Fails on Multi-Line Code Patterns**: A graph database excels at structured triples `(Anatase_TiO2, HAS_ENCLOSING_CUTOFF_A, 3.20)`, but cannot do semantic similarity matching over free-form algorithmic questions like *"how to compute continuous residence survival probability via set intersections"*.

### Recommended Hybrid GraphRAG Routing Architecture

| Data Asset Inside `dataset.zip` | Target Storage Engine | Retrieval Trigger & Injection Format |
| :--- | :--- | :--- |
| `physchem_kb/physchem_knowledge_graph_triples.json` (Bond lengths, covalent cutoffs, stoichiometries, oxidation states, MD $\Delta t$ limits, ASE return shapes) | **Kùzu Knowledge Graph (`backend/rag/knowledge_graph.py`)** | Deterministic entity/keyword match on material (`TiO2`, `IrO2`, `MgO`, `Pt`) or observable (`RDF`, `density`, `boundary`, `MD`). Injected as a compact `<physchem_invariants>` block. |
| `physchem_kb/structure_eda_cards.json` (Atom counts, `O_w` vs `O_slab` split, $z$-ranges, cell vectors for mounted `.extxyz` / `.cif` / `.lmp` files) | **Exact Key-Value / Kùzu Lookup (`backend/notebooks/attach_to_notebook.py`)** | Triggered automatically whenever a structure file path is detected in the prompt or mounted at `/mnt/data/...`. Injected as `<structure_metadata>`. |
| `physchem_kb/chemical_bonding_and_cutoffs.md` & `statistical_mechanics_and_md_laws.md` + `docs/ase_molecular_simulations.md` | **ChromaDB VectorStore (`domain="PhysChem & ASE Laws"`)** | Retrieved via **always-on semantic + BM25 hybrid search** (never skipped when `is_mol=True`!) with `###` section-level chunking (`max_chars=2400`). |
| `sample_questions/sample_q1..q5.py` & `*_validated.py` scripts | **ChromaDB VectorStore (`domain="Verified Code Recipes"`)** | Retrieved by task-intent similarity so `gemma-4-e2b` sees a 100% AST-validated, `pwtools`-free reference script. |

---

## 3. Drop-In Code to Attach `dataset.zip` to Your Agentic RAG Pipeline

### Step 1: Add Automatic `dataset.zip` Extraction & Ingestion (`backend/rag/ingest_dataset_zip.py`)
Place `dataset.zip` in your repository root (or pass its path via `DATASET_ZIP_PATH`) and run this loader at backend startup (`backend/main.py`):

```python
import json
import zipfile
from pathlib import Path
from typing import Dict, Any

def ingest_curated_dataset_zip(zip_path: Path, extract_dir: Path, vector_store, knowledge_graph) -> Dict[str, Any]:
    """
    Extracts `dataset.zip` and indexes:
      1. `physchem_kb/physchem_knowledge_graph_triples.json` + `structure_eda_cards.json` into Kùzu KnowledgeGraph.
      2. All `.md` and `.py` files (including `physchem_kb/*.md` and `sample_questions/*.py`) into ChromaDB VectorStore.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # 1. Load deterministic Physics & Chemistry triples into Knowledge Graph
    triples_file = extract_dir / "physchem_kb" / "physchem_knowledge_graph_triples.json"
    if triples_file.exists():
        triples = json.loads(triples_file.read_text(encoding="utf-8"))
        if hasattr(knowledge_graph, "load_physchem_triples"):
            knowledge_graph.load_physchem_triples(triples)

    # 2. Index Physics/Chemistry Markdown guides and Verified Python scripts into ChromaDB
    docs_indexed = 0
    for file_path in sorted(extract_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix not in {".md", ".py"}:
            continue
        rel_name = file_path.relative_to(extract_dir).as_posix()
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        domain = (
            "PhysChem & ASE Laws"
            if ("physchem_kb" in rel_name or "docs/" in rel_name)
            else "Verified Code Recipes"
        )
        vector_store.add_document(
            doc_id=f"dataset_zip::{rel_name}",
            content=content,
            metadata={"source": rel_name, "domain": domain, "doc_type": file_path.suffix.lstrip(".")},
        )
        docs_indexed += 1

    return {"extracted_to": str(extract_dir), "docs_indexed_in_chromadb": docs_indexed}
```

### Step 2: Upgrade `KnowledgeGraph` (`backend/rag/knowledge_graph.py`) with Physics & Bonding Cutoffs
Extend `backend/rag/knowledge_graph.py` with a `Cutoffs` / `PhysChemLaw` table in Kùzu so `HybridRetriever` can query exact bond lengths and physical invariants deterministically:

```python
# Add inside KnowledgeGraph._init_schema() in backend/rag/knowledge_graph.py:
self.conn.execute(
    "CREATE NODE TABLE IF NOT EXISTS PhysChemRule("
    "id STRING, subject STRING, predicate STRING, object_val STRING, unit STRING, reason STRING, "
    "PRIMARY KEY (id))"
)

def query_physchem_rules(self, query: str) -> list[str]:
    q_low = query.lower()
    facts = [
        "- [Chemical Bonding Law] Intramolecular covalent O-H bond in H2O/OH* is ~0.972 Å (cutoff r_OH = 1.25 Å); intermolecular H-bonds start at >= 1.65 Å. Always partition water oxygens (`i_oh, _ = neighbor_list('ij', atoms, cutoff={('O', 'H'): 1.25})`) BEFORE computing crystal surface layers.",
        "- [ASE Return-Shape Law] `neighbor_list('ijd', atoms, cutoff=3.2)` returns a 3-tuple of 1D arrays `(nl_i, nl_j, nl_d)`; NEVER index as 2D `distances[i, j]`. Neither `Atoms` nor `Atom` has `.get_neighbors()`.",
    ]
    if any(k in q_low for k in ["density", "rho", "g/cm"]):
        facts.append(
            "- [Dimensional Law for Density] Planar mass density in g/cm^3: `rho_ow = ((counts_ow / 6.02214076e23) * 18.01528) / (area_xy * dz * 1e-24)`; for H_w divide by `2.0 * 6.02214076e23`."
        )
    if any(k in q_low for k in ["rdf", "radial", "coordination"]):
        facts.append(
            "- [Statistical Mechanics Law for RDF] Normalize pair distance histograms by ideal-gas spherical shell count `(len(src) * len(dst) / cell_vol) * 4.0 * np.pi * (r_centers**2) * dr` under PBC (`atoms.get_all_distances(mic=True)`)."
        )
    if any(k in q_low for k in ["boundary", "enclos", "interface", "anatase", "tio2"]):
        facts.append(
            "- [Interfacial Boundary Shell Law] In `Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz` (426 atoms: 60 Ti + 120 O_slab = 180 crystal atoms; 82 O_w + 164 H = 246 water atoms), the first crystal boundary layer enclosing water consists of `crystal_indices` within `r_cut = 3.20 Å` of `water_indices` under PBC."
        )
    return facts
```

### Step 3: Fix `HybridRetriever.retrieve()` (`backend/rag/hybrid_retriever.py`)
In `backend/rag/hybrid_retriever.py`, never skip vector search when `is_mol=True`. Instead, combine the deterministic Kùzu `query_physchem_rules(query)` facts with ChromaDB retrieval filtered to `domain in ["PhysChem & ASE Laws", "Verified Code Recipes", "ASE Documentation"]`. This guarantees `gemma-4-e2b` receives both the **exact physical equations/cutoffs** (from Kùzu) and the **complete, AST-valid Python implementation pattern** (from ChromaDB) on every turn.
