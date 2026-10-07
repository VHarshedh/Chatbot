from typing import Optional
from backend.rag.vectorstore import VectorStore
from backend.rag.knowledge_graph import KnowledgeGraph
from backend.rag.chat_memory import get_chat_memory_store

class HybridRetriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.kg = KnowledgeGraph()
        self.chat_memory = get_chat_memory_store()
        
    def retrieve(self, query: str, current_session_id: Optional[int] = None) -> str:
        # 1. Semantic search across notebook cells and repository documents
        vs_results = self.vector_store.semantic_search(query, n_results=6)
        
        # 2. Extract potential entities and domain relationships from query
        kg_context = ""
        q_lower = query.lower()
        if "pt" in q_lower or "platinum" in q_lower:
            res = self.kg.graph_query("MATCH (m:Material {symbol: 'Pt'})-[:has_pseudo]->(p:Pseudopotential) RETURN p.filename")
            if res:
                kg_context += f"Knowledge Graph fact: The recommended pseudopotential for Pt is {res[0][0]}.\n"
                
        if "water" in q_lower or "h2o" in q_lower:
            res = self.kg.graph_query("MATCH (s:System {name: 'Water'}) RETURN s.type_map, s.temp")
            if res:
                kg_context += f"Knowledge Graph fact: The Water system uses type_map {res[0][0]} and temperature {res[0][1]}K.\n"

        from backend.notebooks.attach_to_notebook import is_anatase_water_molecular_prompt, is_anatase_tio2_prompt
        is_mol = is_anatase_water_molecular_prompt(query)

        if not is_mol and any(w in q_lower for w in ["density", "rdf", "survival", "interfacial", "deepmd", "dp_models", "neighbor"]):
            kg_context += (
                "Domain Ground Truth:\n"
                "- Grounded target notebook: `ChatBot/backend/notebooks/running.ipynb`.\n"
                "- Atomic neighbor analysis & surface cutoffs: MUST use `from ase.neighborlist import neighbor_list, natural_cutoffs`.\n"
                "- Trained DeepMD potential models: located at `Training/DP_Models/DPMD_Water/1/`, `Training/DP_Models/DPMD_Water/2/`, and `Training/DP_Models/DPMD_Water/3/`.\n"
                "- Multi-model averaging: loop over models `list(range(1, dp + 1))` to compute cross-model mean and standard deviation.\n"
            )

        if is_mol:
            # File data must NOT be sent; notebook data must NOT be sent; only provide file location
            kg_context += (
                "Knowledge Graph Fact: Anatase TiO2 Molecular Water Trajectory File:\n"
                "- File Location: `ChatBot/public/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz` (Mounted read-only in Docker sandbox at `/mnt/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz`).\n"
                "- CRITICAL ASE SYNTAX: `read(path, index=':')` returns a Python list of Atoms (`list[ase.Atoms]`). "
                "You MUST unpack the frame (`atoms = traj[0] if isinstance(traj, list) else traj`) before calling `atoms.get_positions()`. A list has no `.get_positions()` attribute.\n"
                "- Physics & Chemistry: Anatase TiO2 slab with molecularly adsorbed water. The z-axis is the surface normal. "
                "Features 5-fold coordinated Ti (Ti_5c CUS), bridging oxygens (O_br), and adsorbed water. Use `ase.neighborlist` for interfacial site cutoffs.\n"
                "- Boundary Extraction: Never compare float coordinates with `==`. For slab boundaries along z, use layer tolerance `tol = 0.8` Å (`z <= z_min + tol` or `z >= z_max - tol`). For box boundaries, use fractional coordinates `scaled = atoms.get_scaled_positions(wrap=True)`.\n"
            )
        elif is_anatase_tio2_prompt(query):
            kg_context += (
                "Knowledge Graph Fact: Anatase TiO2 Interface Notebook:\n"
                "- Grounded target notebook: `ChatBot/backend/notebooks/running.ipynb`.\n"
            )
                
        # 3. Cross-Session Chat Memory RAG Retrieval
        chat_memory_context = ""
        try:
            memories = self.chat_memory.search_memory(
                query=query,
                n_results=2,
                current_session_id=current_session_id,
                max_distance=1.05
            )
            if memories:
                chat_memory_context = self.chat_memory.format_memory_context(memories)
        except Exception as e:
            print(f"Notice: Cross-session memory retrieval skipped: {e}")

        # 4. Fuse results (suppress notebook cells if molecular extxyz requested)
        context = ""
        if not is_mol:
            context += "--- Vector Store Context ---\n"
            for i, res in enumerate(vs_results):
                context += f"Result {i+1}:\n{res['content']}\n\n"
            
        context += "--- Knowledge Graph & Domain Context ---\n"
        context += kg_context

        if chat_memory_context:
            context += f"\n{chat_memory_context}\n"
        
        return context

