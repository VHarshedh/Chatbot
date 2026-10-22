import kuzu
import os
from backend.config import settings

class KnowledgeGraph:
    _instance = None

    def __init__(self):
        # Initialize Kuzu Database
        self.db = kuzu.Database(settings.KUZU_DB_DIR)
        self.conn = kuzu.Connection(self.db)
        self._init_schema()
        self.init_domain_data()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def _init_schema(self):
        try:
            # Nodes
            self.conn.execute("CREATE NODE TABLE IF NOT EXISTS Material (symbol STRING, cif_path STRING, name STRING, stoich_o INT64, PRIMARY KEY (symbol))")
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
        # Upgrade existing databases without dropping their material relationships.
        for column, kind in (("name", "STRING"), ("stoich_o", "INT64")):
            try:
                self.conn.execute(f"ALTER TABLE Material ADD {column} {kind}")
            except RuntimeError as exc:
                if "already exists" not in str(exc).lower() and "already has" not in str(exc).lower():
                    raise
        materials = [
            ("Pt", "./DFT/Pt.cif", "Platinum", 0),
            ("Ir", "./Analysis/iro2_water_concat.extxyz", "Iridium Dioxide (IrO2)", 2),
            ("Ti", "./public/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz", "Anatase Titanium Dioxide (TiO2)", 2),
            ("C", "", "Graphene", 0),
        ]
        for symbol, path, name, stoich in materials:
            self.conn.execute("MERGE (m:Material {symbol: $symbol}) SET m.cif_path=$path, m.name=$name, m.stoich_o=$stoich", {"symbol": symbol, "path": path, "name": name, "stoich": stoich})
        self.conn.execute("MERGE (p:Pseudopotential {filename: 'Pt_ONCV_PBE-1.2.upf'}) SET p.type='ONCV', p.xc='PBE'")
        self.conn.execute("MATCH (m:Material {symbol: 'Pt'}), (p:Pseudopotential {filename: 'Pt_ONCV_PBE-1.2.upf'}) MERGE (m)-[:has_pseudo]->(p)")
        import json
        for name, types, temp in [("Water", ["H", "O"], 300.0), ("Interface", ["O", "H", "C"], 330.0), ("IrO2_Water", ["Ir", "O", "H"], 300.0), ("Anatase_TiO2_Water", ["Ti", "O", "H"], 300.0)]:
            self.conn.execute("MERGE (s:System {name: $name}) SET s.type_map=$types, s.temp=$temp", {"name": name, "types": json.dumps(types), "temp": temp})
        for symbol, system in [("Ir", "IrO2_Water"), ("Ti", "Anatase_TiO2_Water"), ("C", "Interface")]:
            self.conn.execute("MATCH (m:Material {symbol: $symbol}), (s:System {name: $system}) MERGE (m)-[:part_of]->(s)", {"symbol": symbol, "system": system})

    def graph_query(self, query: str) -> list[dict]:
        results = []
        try:
            result = self.conn.execute(query)
            while result.has_next():
                results.append(result.get_next())
        except Exception as e:
            print(f"Query error: {e}")
        return results

def get_knowledge_graph() -> KnowledgeGraph:
    return KnowledgeGraph.get_instance()


if __name__ == "__main__": # pragma: no cover
    kg = KnowledgeGraph()
    kg.init_domain_data()
