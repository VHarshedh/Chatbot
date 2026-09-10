import kuzu
import os
from backend.config import settings

class KnowledgeGraph:
    def __init__(self):
        # Initialize Kuzu Database
        self.db = kuzu.Database(settings.KUZU_DB_DIR)
        self.conn = kuzu.Connection(self.db)
        self._init_schema()
        
    def _init_schema(self):
        try:
            # Nodes
            self.conn.execute("CREATE NODE TABLE IF NOT EXISTS Material (symbol STRING, cif_path STRING, PRIMARY KEY (symbol))")
            self.conn.execute("CREATE NODE TABLE IF NOT EXISTS Pseudopotential (filename STRING, type STRING, xc STRING, PRIMARY KEY (filename))")
            self.conn.execute("CREATE NODE TABLE IF NOT EXISTS System (name STRING, type_map STRING, temp DOUBLE, PRIMARY KEY (name))")
            
            # Relationships
            self.conn.execute("CREATE REL TABLE IF NOT EXISTS has_pseudo (FROM Material TO Pseudopotential)")
            self.conn.execute("CREATE REL TABLE IF NOT EXISTS part_of (FROM Material TO System)")
        except RuntimeError as e:
            # Table might already exist if we catch it, ignore
            if "already exists" not in str(e).lower():
                print(f"Warning: {e}")
                
    def init_domain_data(self):
        # Insert initial domain data
        try:
            self.conn.execute("MERGE (m:Material {symbol: 'Pt', cif_path: './DFT/Pt.cif'})")
            self.conn.execute("MERGE (p:Pseudopotential {filename: 'Pt_ONCV_PBE-1.2.upf', type: 'ONCV', xc: 'PBE'})")
            self.conn.execute("MATCH (m:Material {symbol: 'Pt'}), (p:Pseudopotential {filename: 'Pt_ONCV_PBE-1.2.upf'}) MERGE (m)-[:has_pseudo]->(p)")
            
            self.conn.execute("MERGE (s:System {name: 'Water', type_map: '[\"H\", \"O\"]', temp: 300.0})")
            print("Domain data initialized.")
        except Exception as e:
            print(f"Error inserting domain data: {e}")
            
    def graph_query(self, query: str) -> list[dict]:
        results = []
        try:
            result = self.conn.execute(query)
            while result.has_next():
                results.append(result.get_next())
        except Exception as e:
            print(f"Query error: {e}")
        return results

if __name__ == "__main__": # pragma: no cover
    kg = KnowledgeGraph()
    kg.init_domain_data()
