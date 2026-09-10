from backend.rag.vectorstore import VectorStore
from backend.rag.knowledge_graph import KnowledgeGraph

class HybridRetriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.kg = KnowledgeGraph()
        
    def retrieve(self, query: str) -> str:
        # 1. Semantic search
        vs_results = self.vector_store.semantic_search(query, n_results=3)
        
        # 2. Extract potential entities from query and query KG
        # For a production system, use an LLM or NER to extract entities. 
        # Here we do a simple heuristic for demonstration.
        kg_context = ""
        if "Pt" in query or "platinum" in query.lower():
            res = self.kg.graph_query("MATCH (m:Material {symbol: 'Pt'})-[:has_pseudo]->(p:Pseudopotential) RETURN p.filename")
            if res:
                kg_context += f"Knowledge Graph fact: The recommended pseudopotential for Pt is {res[0][0]}.\n"
                
        if "Water" in query or "water" in query.lower():
            res = self.kg.graph_query("MATCH (s:System {name: 'Water'}) RETURN s.type_map, s.temp")
            if res:
                kg_context += f"Knowledge Graph fact: The Water system uses type_map {res[0][0]} and temperature {res[0][1]}K.\n"
                
        # 3. Fuse results
        context = "--- Vector Store Context ---\n"
        for i, res in enumerate(vs_results):
            context += f"Result {i+1}:\n{res['content']}\n\n"
            
        context += "--- Knowledge Graph Context ---\n"
        context += kg_context
        
        return context
