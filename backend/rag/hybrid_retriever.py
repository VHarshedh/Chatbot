from backend.rag.vectorstore import VectorStore
from backend.rag.knowledge_graph import KnowledgeGraph

class HybridRetriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.kg = KnowledgeGraph()
        
    def retrieve(self, query: str) -> str:
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

        if any(w in q_lower for w in ["density", "rdf", "survival", "interfacial", "deepmd", "dp_models", "iro2", "neighbor"]):
            kg_context += (
                "Domain Ground Truth:\n"
                "- Reference implementation notebook: `Analysis/Water_Density_RDF_Survival_Probability_Hydroxyl_Fraction_with_IrO2_Template_Seperate_Plot_Capability.ipynb`.\n"
                "- Atomic neighbor analysis & surface cutoffs: MUST use `from ase.neighborlist import neighbor_list, natural_cutoffs`.\n"
                "- Trained DeepMD potential models: located at `Training/DP_Models/DPMD_Water/1/`, `Training/DP_Models/DPMD_Water/2/`, and `Training/DP_Models/DPMD_Water/3/`.\n"
                "- Multi-model averaging: loop over models `list(range(1, dp + 1))` to compute cross-model mean and standard deviation.\n"
            )
                
        # 3. Fuse results
        context = "--- Vector Store Context ---\n"
        for i, res in enumerate(vs_results):
            context += f"Result {i+1}:\n{res['content']}\n\n"
            
        context += "--- Knowledge Graph & Domain Context ---\n"
        context += kg_context
        
        return context

