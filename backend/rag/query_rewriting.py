import re


def expand_query(query):
    """Preserve identifiers while adding lexical equivalents for scientific acronyms."""
    expansions = {"DFT": "density functional theory Quantum Espresso", "RDF": "radial distribution function", "CUS": "coordinatively unsaturated sites", "MD": "molecular dynamics LAMMPS", "DP": "DeepMD deep potential", "SCF": "self consistent field", "PBC": "periodic boundary conditions"}
    terms = [text for acronym, text in expansions.items() if re.search(r"\b" + acronym + r"\b", query, re.I)]
    return query + ("\nSearch terms: " + "; ".join(terms) if terms else "")
