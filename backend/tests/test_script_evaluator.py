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
    assert "reference_notebook" in res
    assert res["notebook_features"] == {}
    assert res["metric_comparison"] is None

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
    assert "reference_notebook" in res
    assert res["notebook_features"]["ase_neighborlist_api"] is True

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

    # Grade C (70-79 pts)
    script_c = """```python
import numpy as np
import ase
import matplotlib.pyplot as plt

def classify_sites(atoms, metal_symbol='Ir'):
    z_coords = atoms.positions[:, 2]
    m_sur = np.where((atoms.symbols == metal_symbol) & (z_coords > np.max(z_coords) - 1.5))[0]
    cus_sites = [m_sur[0]]
    return m_sur, cus_sites

def compute(traj):
    bins_z = np.arange(0, 30, 0.1)
    mass_density = np.zeros(len(bins_z)-1)
    r = np.arange(0, 10, 0.05)
    g_r = np.sin(r) + 1.0
    frac_oh = 0.45
    tau = np.logspace(0, 4, 100)
    p_t = np.exp(-tau / 50.0)
    return frac_oh, tau, p_t

dp_models = [1, 2]
mean_val = np.mean([1, 2])
plt.subplots(2, 2)
```"""
    res_c = evaluator.evaluate(script_c)
    assert res_c["grade"] in ("B", "C", "D")

    # Grade D (60-69 pts)
    script_d = """```python
import numpy as np
import ase
import matplotlib.pyplot as plt

def classify_sites(atoms, metal_symbol='Ir'):
    z_coords = atoms.positions[:, 2]
    m_sur = np.where(atoms.symbols == metal_symbol)[0]
    return m_sur

def compute(traj):
    bins_z = np.arange(0, 30, 0.1)
    mass_density = np.zeros(len(bins_z)-1)
    frac_oh = 0.45
    tau = np.logspace(0, 4, 100)
    p_t = np.exp(-tau / 50.0)
    return frac_oh, tau, p_t

plt.subplots(2, 2)
```"""
    res_d = evaluator.evaluate(script_d)
    assert res_d["grade"] in ("C", "D", "F")

def test_extract_python_code_with_thinking_tags():
    evaluator = ScriptEvaluator()
    response_with_thinking = """<details class="thinking-details"><summary><em>Thinking Process (50 words)</em></summary>
# Thought 1: Let's consider how to write the script
# Thought 2: We need to import ase and numpy
</details>

Here is the implementation:
```python
import numpy as np
import ase
from ase.neighborlist import neighbor_list
print('Compiled code')
```
"""
    code = evaluator.extract_python_code(response_with_thinking)
    assert not code.startswith("<details")
    assert "import numpy as np" in code
    res = evaluator.evaluate(response_with_thinking)
    assert any("Valid Python AST syntax" in p for p in res["passed_checks"])
    assert not any("Syntax error" in f for f in res["failed_checks"])

def test_reference_notebook_and_metrics():
    evaluator = ScriptEvaluator()
    assert "Water_Density_RDF_Survival_Probability" in evaluator.REFERENCE_NOTEBOOK
    assert "tau_oh_ns" in evaluator.REFERENCE_METRICS
    assert "tau_h2o_ns" in evaluator.REFERENCE_METRICS
    assert "frac_oh_pct" in evaluator.REFERENCE_METRICS
    assert "frac_h2o_pct" in evaluator.REFERENCE_METRICS
    assert evaluator.REFERENCE_METRICS["tau_oh_ns"]["value"] == 0.298
    assert evaluator.REFERENCE_METRICS["tau_h2o_ns"]["value"] == 0.043
    assert evaluator.REFERENCE_METRICS["frac_oh_pct"]["value"] == 77.44
    assert evaluator.REFERENCE_METRICS["frac_h2o_pct"]["value"] == 22.56

def test_extract_metrics():
    sample_text = """
    === FINAL REPORT ===
    Residence time tau_OD: 0.2980 ns
    Residence time tau_D2O: 0.0430 ns
    Surface Dissociation (OD): 77.4%
    Molecular fraction (D2O): 22.6%
    """
    metrics = ScriptEvaluator.extract_metrics(sample_text)
    assert metrics["tau_oh_ns"] == 0.298
    assert metrics["tau_h2o_ns"] == 0.043
    assert metrics["frac_oh_pct"] == 77.4
    assert metrics["frac_h2o_pct"] == 22.6

    # Empty text
    assert ScriptEvaluator.extract_metrics("") == {}

def test_compare_metrics():
    evaluator = ScriptEvaluator()
    
    # Matching within 5% tolerance
    cand_pass = {
        "tau_oh_ns": 0.300,   # ~0.67% error
        "tau_h2o_ns": 0.044,  # ~2.3% error
        "frac_oh_pct": 78.0,  # ~0.72% error
        "frac_h2o_pct": 22.0  # ~2.48% error
    }
    comp = evaluator.compare_metrics(cand_pass, tolerance=0.05)
    assert comp["all_passed"] is True
    assert comp["passed_count"] == 4
    assert comp["metrics_accuracy_score"] == 100.0
    assert comp["metrics"]["tau_oh_ns"]["passed"] is True

    # Failing tolerance (e.g. 50% error on tau_oh)
    cand_fail = {
        "tau_oh_ns": 0.600,   # >100% error
        "tau_h2o_ns": 0.043
    }
    comp_fail = evaluator.compare_metrics(cand_fail, tolerance=0.05)
    assert comp_fail["all_passed"] is False
    assert comp_fail["passed_count"] == 1
    assert comp_fail["metrics"]["tau_oh_ns"]["passed"] is False
    assert comp_fail["metrics"]["tau_h2o_ns"]["passed"] is True

def test_evaluate_with_embedded_metrics():
    evaluator = ScriptEvaluator()
    script_with_metrics = """```python
import numpy as np
import ase
from ase.neighborlist import neighbor_list
import matplotlib.pyplot as plt

# Summary of results:
# tau_OD: 0.298 ns
# tau_D2O: 0.043 ns
# Dissociated: 77.4%
print("Analysis complete")
```"""
    res = evaluator.evaluate(script_with_metrics)
    assert res["metric_comparison"] is not None
    assert res["metric_comparison"]["all_passed"] is True
    assert res["metric_comparison"]["passed_count"] >= 2
