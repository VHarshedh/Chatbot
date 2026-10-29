import kuzu
import os
import json
import hashlib
import re
from backend.config import settings

class KnowledgeGraph:
    _instance = None

    def __init__(self):
        # Initialize Kuzu Database with read-only fallback on concurrent lock
        try:
            self.db = kuzu.Database(settings.KUZU_DB_DIR)
            self.conn = kuzu.Connection(self.db)
            self._init_schema()
            self.init_domain_data()
        except RuntimeError as e:
            if "lock" in str(e).lower():
                try:
                    self.db = kuzu.Database(settings.KUZU_DB_DIR, read_only=True)
                    self.conn = kuzu.Connection(self.db)
                except Exception:
                    self.db = None
                    self.conn = None
            else:
                raise

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
            self.conn.execute("CREATE NODE TABLE IF NOT EXISTS PhysChemRule(id STRING, subject STRING, predicate STRING, object_val STRING, unit STRING, reason STRING, PRIMARY KEY(id))")
            self.conn.execute("CREATE NODE TABLE IF NOT EXISTS StructureCard(path STRING, payload STRING, PRIMARY KEY(path))")
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

    def load_physchem_triples(self, triples):
        for row in triples:
            subject, predicate, value = str(row['subject']), str(row['predicate']), str(row['object'])
            key = hashlib.sha256(f'{subject}\0{predicate}'.encode()).hexdigest()
            details = {k: v for k, v in row.items() if k not in {'subject', 'predicate', 'object', 'unit'}}
            self.conn.execute(
                'MERGE (r:PhysChemRule {id: $id}) SET r.subject=$subject, r.predicate=$predicate, r.object_val=$value, r.unit=$unit, r.reason=$reason',
                {'id': key, 'subject': subject, 'predicate': predicate, 'value': value,
                 'unit': str(row.get('unit', '')), 'reason': json.dumps(details, ensure_ascii=False)})

    def load_structure_cards(self, cards):
        for path, card in cards.items():
            self.conn.execute('MERGE (c:StructureCard {path: $path}) SET c.payload=$payload',
                              {'path': path, 'payload': json.dumps(card, ensure_ascii=False)})

    def structure_metadata(self, query):
        """Match explicit file names/paths, never transfer counts between materials."""
        result = self.conn.execute('MATCH (c:StructureCard) RETURN c.path, c.payload') if self.conn else None
        cards = {}
        while result and result.has_next():
            path, payload = result.get_next()
            card = json.loads(payload)
            names = [path, card.get('fallback_path', ''), path.rsplit('/', 1)[-1]]
            if any(name and re.search(r'(?<![\w.-])' + re.escape(name) + r'(?![\w.-])', query, re.I) for name in names):
                cards[path] = card
        return cards

    def query_physchem_rules(self, query):
        q = query.lower().replace('_', ' ')
        aliases = {'Anatase_TiO2': ('tio2', 'anatase', 'titanium'),
                   'Rutile_IrO2': ('iro2', 'iridium'), 'Rocksalt_MgO': ('mgo', 'magnesium'),
                   'Quartz_SiO2': ('sio2', 'quartz', 'silica')}
        scientific = any(w in q for w in ('water', 'h2o', 'ase', 'neighbor', 'rdf', 'density', 'boundary', 'interface', 'hydroxyl', 'molecular dynamics')) or any(w in q for words in aliases.values() for w in words)
        facts = []
        result = self.conn.execute('MATCH (r:PhysChemRule) RETURN r.subject, r.predicate, r.object_val, r.unit, r.reason ORDER BY r.id') if self.conn else None
        while result and result.has_next():
            subject, predicate, value, unit, reason = result.get_next()
            words = aliases.get(subject, (subject.lower(),))
            match = any(re.search(r'(?<!\w)' + re.escape(w) + r'(?!\w)', q) for w in words)
            if subject in {'H2O', 'ASE_neighbor_list_ijd'}:
                match = scientific
            if subject == 'H2O_Flexible_MD':
                match = scientific and bool(re.search(r'\b(md|timestep|langevin|verlet)\b|molecular dynamics', q))
            if match:
                facts.append(f'- {subject}: {predicate} = {value} {unit}. Details: {reason}')
        if scientific:
            facts.append("- Partition water/hydroxyl oxygens with covalent O-H neighbors before selecting crystal layers. neighbor_list('ijd', atoms, cutoff=...) returns three 1D arrays (i, j, d), not a distance matrix; Atom/Atoms has no get_neighbors().")
        if any(w in q for w in ('density', 'rho', 'g/cm')):
            facts.append('- Density (g/cm^3): rho_ow = ((counts_ow / 6.02214076e23) * 18.01528) / (area_xy * dz * 1e-24); for H_w divide counts by 2. area_xy = norm(cross(cell[0], cell[1])).')
        if any(w in q for w in ('rdf', 'radial distribution', 'coordination')):
            facts.append('- RDF: normalize pair histograms by (N_A * N_B / V) * (4*pi/3) * (r_outer**3 - r_inner**3), exclude self pairs, and use periodic minimum-image distances within the valid box radius.')
        return facts

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
