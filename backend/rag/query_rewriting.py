import re


def expand_query(query):
    """Preserve identifiers while adding lexical equivalents for scientific acronyms."""
    expansions = {"DFT": "density functional theory Quantum Espresso", "RDF": "radial distribution function", "CUS": "coordinatively unsaturated sites", "MD": "molecular dynamics LAMMPS", "DP": "DeepMD deep potential", "SCF": "self consistent field", "PBC": "periodic boundary conditions"}
    terms = [text for acronym, text in expansions.items() if re.search(r"\b" + acronym + r"\b", query, re.I)]
    return query + ("\nSearch terms: " + "; ".join(terms) if terms else "")


def decompose_query(query):
    """Split explicit multi-part requests while retaining shared material context."""
    clauses = [part.strip() for part in re.split(r";|\n(?=\d+[.)]\s)", query) if part.strip()]
    if len(clauses) <= 1:
        return [query]
    materials = " ".join(dict.fromkeys(re.findall(r"\b(?:TiO2|IrO2|anatase|graphene|platinum|water)\b", query, re.I)))
    return [part + " " + materials for part in clauses[:3]]
