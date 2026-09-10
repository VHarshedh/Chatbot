import pytest
from backend.evaluation.script_evaluator import ScriptEvaluator

def test_extract_python_code():
    evaluator = ScriptEvaluator()

    # Markdown block
    markdown = "Here is the code:\n```python\nimport numpy as np\nprint('hello')\n```\nHope it helps!"
    assert evaluator.extract_python_code(markdown) == "import numpy as np\nprint('hello')"

    # Fallback to plain text if code-like
    plain = "import ase\nimport numpy as np\ndef run(): pass"
    assert evaluator.extract_python_code(plain) == plain

    # Empty text
    assert evaluator.extract_python_code("") == ""
    assert evaluator.extract_python_code("Just some natural language explanation without code.") == ""

def test_evaluate_empty():
    evaluator = ScriptEvaluator()
    res = evaluator.evaluate("")
    assert res["score"] == 0.0
    assert res["grade"] == "F"
    assert len(res["failed_checks"]) == 1

def test_evaluate_syntax_error():
    evaluator = ScriptEvaluator()
    bad_code = "```python\nimport ase\ndef invalid_syntax(\n```"
    res = evaluator.evaluate(bad_code)
    assert any("Syntax error" in f for f in res["failed_checks"])

def test_evaluate_full_ideal_script():
    evaluator = ScriptEvaluator()

    ideal_script = """```python
import numpy as np
import ase
from ase.io import read
from ase.neighborlist import neighbor_list
import matplotlib.pyplot as plt

# 1. Site identification
def classify_sites(atoms, metal_symbol='Ir', o_stoich=2):
    z_coords = atoms.positions[:, 2]
    m_sur = np.where((atoms.symbols == metal_symbol) & (z_coords > np.max(z_coords) - 1.5))[0]
    o_indices = np.where(atoms.symbols == 'O')[0]
    o_bulk = o_indices[:len(m_sur) * o_stoich]
    o_w = [i for i in o_indices if i not in o_bulk]
    
    # CUS metal sites and adsorbed oxygen
    cus_sites = []
    o_adb = []
    for m in m_sur:
        # Check water oxygen within 2.5 A
        cus_sites.append(m)
    o_br = [i for i in o_bulk[-len(cus_sites):]] # bridge oxygens
    return m_sur, o_bulk, o_w, cus_sites, o_adb, o_br

# 2. Water density profiles & Pair RDFs
def compute_density_and_rdf(trajectories):
    bins_z = np.arange(0, 30, 0.1)
    mass_density = np.zeros(len(bins_z)-1) # 18.015 g/mol
    r = np.arange(0, 10, 0.05)
    g_r = np.sin(r) + 1.0 # RDF pair_distribution
    return bins_z, mass_density, r, g_r

# 3. Surface Kinetics & Survival Probability
def compute_kinetics(traj, o_adb):
    # Hydroxyl fraction (OH vs intact H2O via neighbor_list cutoff 1.2 A)
    frac_oh = 0.45 # dissociated fraction
    # Survival probability P(t) over time_lag tau
    tau = np.logspace(0, 4, 100)
    p_t = np.exp(-tau / 50.0) # residence intersection
    return frac_oh, tau, p_t

# 4. Multi-model averaging across DP models & Artifacts
def main():
    dp_models = [1, 2, 3, 4]
    densities = []
    for model_idx in dp_models:
        pass
    mean_density = np.mean(densities, axis=0) if densities else np.zeros(10)
    std_density = np.std(densities, axis=0) if densities else np.zeros(10)

    # Individual plots
    plt.figure()
    plt.plot([1, 2], [3, 4])
    plt.savefig("density_profile.png")

    # Combined 2x2 multi-panel summary figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("2x2 summary")
    plt.savefig("full_analysis_presentation.png")

    # Text summary report
    with open("summary_report.txt", "w") as f:
        f.write("residence_time: 15.2 ps\\n")
```"""

    res = evaluator.evaluate(ideal_script)
    assert res["score"] == 100.0
    assert res["grade"] == "A"
    assert len(res["passed_checks"]) == 16
    assert len(res["failed_checks"]) == 0
    assert res["category_scores"]["syntax_and_imports"]["score"] == 15
    assert res["category_scores"]["site_identification"]["score"] == 20
    assert res["category_scores"]["density_and_rdfs"]["score"] == 20
    assert res["category_scores"]["kinetics_and_survival"]["score"] == 25
    assert res["category_scores"]["model_averaging_and_outputs"]["score"] == 20

def test_evaluate_partial_script():
    evaluator = ScriptEvaluator()
    partial = """```python
import numpy as np
import matplotlib.pyplot as plt

# Just basic density calculation
bins_z = np.arange(0, 20, 0.1)
density = np.histogram(bins_z)[0]
plt.plot(bins_z[:-1], density)
plt.savefig("plot.png")
```"""
    res = evaluator.evaluate(partial)
    assert 10.0 <= res["score"] <= 50.0
    assert res["grade"] in ("D", "F")
    assert len(res["failed_checks"]) > 0

def test_evaluate_grades():
    evaluator = ScriptEvaluator()
    # Grade B test script (80-89 pts)
    script_b = """```python
import numpy as np
import ase
from ase.io import read
from ase.neighborlist import neighbor_list
import matplotlib.pyplot as plt

def classify_sites(atoms, metal_symbol='Ir', o_stoich=2):
    z_coords = atoms.positions[:, 2]
    m_sur = np.where((atoms.symbols == metal_symbol) & (z_coords > np.max(z_coords) - 1.5))[0]
    o_indices = np.where(atoms.symbols == 'O')[0]
    o_bulk = o_indices[:len(m_sur) * o_stoich]
    o_w = [i for i in o_indices if i not in o_bulk]
    cus_sites = [m_sur[0]]
    o_adb = []
    o_br = []
    return m_sur, o_bulk, o_w, cus_sites, o_adb, o_br

def compute(traj):
    bins_z = np.arange(0, 30, 0.1)
    mass_density = np.zeros(len(bins_z)-1) # 18.015 g/mol
    r = np.arange(0, 10, 0.05)
    g_r = np.sin(r) + 1.0
    frac_oh = 0.45
    tau = np.logspace(0, 4, 100)
    p_t = np.exp(-tau / 50.0)
    return frac_oh, tau, p_t

dp_models = [1, 2]
mean_val = np.mean([1, 2])
std_val = np.std([1, 2])
plt.subplots(2, 2)
```"""
    res_b = evaluator.evaluate(script_b)
    assert res_b["grade"] in ("A", "B", "C")
