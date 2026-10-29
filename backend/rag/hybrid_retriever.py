import re
from typing import Optional, List, Dict, Any
from backend.rag.vectorstore import VectorStore
from backend.rag.knowledge_graph import KnowledgeGraph
from backend.rag.chat_memory import get_chat_memory_store
from backend.notebooks.attach_to_notebook import (
    is_anatase_water_molecular_prompt,
    is_anatase_tio2_prompt,
)

class HybridRetriever:
    _vs_instance: Optional[VectorStore] = None
    _kg_instance: Optional[KnowledgeGraph] = None

    def __init__(self):
        if HybridRetriever._vs_instance is None:
            HybridRetriever._vs_instance = VectorStore()
        if HybridRetriever._kg_instance is None:
            HybridRetriever._kg_instance = KnowledgeGraph.get_instance()
        self.vector_store = HybridRetriever._vs_instance
        self.kg = HybridRetriever._kg_instance
        self.chat_memory = get_chat_memory_store()

    @staticmethod
    def _detect_domain_filter(query: str) -> Optional[Dict[str, Any]]:
        q = query.lower()
        if any(w in q for w in ["ecutwfc", "ecutrho", "kpoints", "k-point", "pw.in", "quantum espresso", "pseudopotential"]):
            return {"domain": "DFT"}
        if any(w in q for w in ["in.lammps", "read_restart", "write_restart", "fix 1 all nvt", "submit_from.sh"]):
            return {"domain": "Molecular Dynamics"}
        if any(w in q for w in ["coord.raw", "box.raw", "raw_to_set", "se_e2_a", "start_pref_e", "lcurve.out"]):
            return {"domain": "DeepMD Training"}
        if any(w in q for w in ["survival probability", "cus", "o_adb", "o_br", "hydroxyl", "residence time", "simulation_results.npz"]):
            return {"domain": "Interfacial Analysis"}
        if any(w in q for w in ["ase", "neighbor_list", "atoms.", "get_positions", "get_distance", "natural_cutoffs", "velocityverlet", "langevin"]):
            return {"domain": "ASE Documentation"}
        return None

    @staticmethod
    def _compress_chunk(content: str, max_chars: int = 1400) -> str:
        """Strips verbose decorative comment banners from notebook cells to save context window tokens."""
        lines = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#" * 10):
                continue
            lines.append(line)
        cleaned = "\n".join(lines).strip()
        return cleaned[:max_chars] + ("\n# ... [truncated]" if len(cleaned) > max_chars else "")

    def retrieve(self, query: str, current_session_id: Optional[int] = None, max_chunks: int = 4) -> str:
        q_lower = query.lower()
        is_mol = is_anatase_water_molecular_prompt(query)

        # 1. Structured Knowledge Graph & Domain Facts
        kg_facts = self.kg.linked_facts(query)

        if is_mol:
            kg_facts.extend([
                "- Trajectory File Path: `/mnt/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz` (or `ChatBot/public/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz`).",
                "- ASE Trajectory Unpacking: `traj = read(path, index=':'); atoms = traj[0] if isinstance(traj, list) else traj`.",
                "- Surface Layer Selection: Use tolerance window `tol = 0.8` Å (`pos[:, 2] <= z_min + tol` or `pos[:, 2] >= z_max - tol`), never float `==`.",
                "- Coordination & Cutoffs: Use `from ase.neighborlist import neighbor_list, natural_cutoffs`."
            ])
        if any(w in q_lower for w in ["ase", "neighbor_list", "atomic simulation environment", "trajectory", "atoms"]):
            kg_facts.extend([
                "- ASE Single Frame vs Trajectory: `atoms = read(path, index=0)` returns `Atoms`; `traj = read(path, index=':')` returns `list[Atoms]` (must unpack before `.get_positions()`).",
                "- ASE Neighbor Finding: `neighbor_list('ijd', atoms, cutoff=...)` returns 3 1D arrays `(i, j, d)`; pair-specific cutoffs: `neighbor_list('ij', atoms, cutoff={('O', 'H'): 1.25})`.",
                "- Individual Atom vs Atoms: An individual `atom` from `for atom in atoms` has `.position` (shape (3,)), NOT `.get_positions()` or `.get_neighbors()`.",
                "- Periodic Minimum Image Convention (MIC): `atoms.get_distance(i, j, mic=True)`, `atoms.get_all_distances(mic=True)`, `atoms.get_angle(a1, a2, a3, mic=True)`.",
                "- Molecular Dynamics Integrators: `Langevin(atoms, timestep=dt, temperature_K=T, friction=gamma)`, `VelocityVerlet(atoms, timestep=dt)`.",
                "- Geometry Optimization: `BFGS(atoms, trajectory='opt.traj').run(fmax=0.02)`."
            ])

        # 2. Always retrieve code and laws, including molecular structure requests.
        invariants = self.kg.query_physchem_rules(query)
        scientific = is_mol or bool(invariants)
        domain_filter = self._detect_domain_filter(query)
        if scientific:
            domains = ["PhysChem & ASE Laws", "Verified Code Recipes", "ASE Documentation"]
            if domain_filter:
                domains.append(domain_filter["domain"])
            domain_filter = {"domain": {"$in": domains}}
        from backend.rag.query_rewriting import expand_query, decompose_query
        from backend.rag.vectorstore import reciprocal_rank_fusion
        searches = [self.vector_store.semantic_search(expand_query(part), n_results=max_chunks, filter=domain_filter)
                    for part in decompose_query(query)] if max_chunks > 0 else []
        vs_results = searches[0] if searches else []
        for results in searches[1:]:
            vs_results = reciprocal_rank_fusion(vs_results, results, top_n=3)
        if not vs_results and domain_filter is not None and max_chunks > 0:
            vs_results = self.vector_store.semantic_search(query, n_results=max_chunks)
        selected_chunks = []
        seen = set()
        for res in vs_results:
            content = res.get("content", "").strip()
            if content and content not in seen:
                seen.add(content)
                siblings = self.vector_store.notebook_siblings(res)
                if not isinstance(siblings, str):
                    siblings = ""
                selected_chunks.append(self._compress_chunk(content, max_chars=900))
                if siblings:
                    selected_chunks.append(siblings)
            if len(selected_chunks) >= max_chunks:
                break

        # 3. Cross-Session Episodic Memory (Strict distance threshold)
        memory_block = ""
        try:
            memories = self.chat_memory.search_memory(
                query=query,
                n_results=2,
                current_session_id=current_session_id,
                max_distance=0.85
            )
            requested_ti = bool(re.search(r"tio2|anatase|titanium", query, re.I))
            requested_ir = bool(re.search(r"iro2|iridium", query, re.I))
            if requested_ti != requested_ir:
                conflict = r"iro2|iridium" if requested_ti else r"tio2|anatase|titanium"
                memories = [m for m in memories if not re.search(conflict, m.get("content", ""), re.I)]
            if memories:
                memory_block = self.chat_memory.format_memory_context(memories)
        except Exception:
            pass

        # 4. Assemble Compact, High-Signal Context for Gemma 4 E2B
        from html import escape
        # Conservative byte budget: UTF-8 bytes upper-bound byte-level token counts.
        # Keep delimiters intact; omit complete blocks that cannot fit.
        sections = ["<retrieved_context>"]
        remaining = 1800 - len("<retrieved_context>\n</retrieved_context>".encode())
        def add(tag, text):
            nonlocal remaining
            if not text:
                return
            block = f"<{tag}>\n{escape(str(text))}\n</{tag}>"
            size = len(block.encode("utf-8")) + 1
            if size <= remaining:
                sections.append(block)
                remaining -= size
        if invariants:
            for fact in invariants[:3]:
                add("physchem_invariants", fact)
        cards = self.kg.structure_metadata(query)
        if cards:
            import json
            add("structure_metadata", json.dumps(cards))
        for chunk in selected_chunks:
            add("reference_chunk", chunk)
        for fact in kg_facts:
            add("knowledge_graph_fact", fact)
        if memory_block:
            add("episodic_memory", memory_block)
        sections.append("</retrieved_context>")
        return "\n".join(sections)
