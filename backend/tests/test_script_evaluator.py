import pytest
from backend.evaluation.script_evaluator import ScriptEvaluator
from backend.evaluation.ast_analyzer import AstAnalyzer
from backend.evaluation.domain_rules import DomainClassifier, DomainRule, ScientificDomain
from backend.evaluation.dynamic_test_generator import DynamicTestGenerator
from backend.evaluation.benchmark_evaluator import BenchmarkEvaluator

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

def test_ast_analyzer_features():
    code = """
import numpy as np
from ase.io import read
import matplotlib.pyplot as plt

def compute_adsorption_energy(slab, adsorbate, e_clean, e_mol):
    \"\"\"Calculates surface adsorption energy in eV.\"\"\"
    e_total = slab.get_potential_energy()
    e_ads = e_total - (e_clean + e_mol)
    return e_ads

fig = plt.figure()
plt.plot([1, 2], [3, 4])
plt.savefig("ads.png")
with open("result.txt", "w") as f:
    f.write("done")
"""
    analyzer = AstAnalyzer(code)
    report = analyzer.analyze()
    assert report["syntax_valid"] is True
    assert "numpy" in report["imports"]["modules"]
    assert "ase.io.read" in report["imports"]["from_imports"]
    assert len(report["functions"]) == 1
    assert report["functions"][0]["name"] == "compute_adsorption_energy"
    assert len(report["functions"][0]["args"]) == 4
    assert report["safety"]["safe"] is True
    assert report["patterns"]["has_plotting"] is True
    assert report["patterns"]["has_saving"] is True
    assert report["patterns"]["has_file_io"] is True
    assert "numpy" in report["patterns"]["scientific_libs"]
    assert "ase" in report["patterns"]["scientific_libs"]

def test_ast_analyzer_syntax_error_and_empty():
    empty_analyzer = AstAnalyzer("")
    assert empty_analyzer.syntax_valid is False
    assert empty_analyzer.extract_imports() == {"modules": set(), "from_imports": set(), "aliases": set()}
    assert empty_analyzer.extract_functions() == []
    assert empty_analyzer.check_safety()["safe"] is True
    assert empty_analyzer.detect_computational_patterns()["loop_count"] == 0

    bad_analyzer = AstAnalyzer("def broken(: pass")
    assert bad_analyzer.syntax_valid is False
    assert bad_analyzer.syntax_error_msg is not None

def test_ast_analyzer_safety_violation():
    unsafe_code = """
import os
os.system("rm -rf /")
eval("print('danger')")
"""
    analyzer = AstAnalyzer(unsafe_code)
    report = analyzer.analyze()
    assert report["safety"]["safe"] is False
    assert len(report["safety"]["violations"]) >= 1

def test_domain_classifier_and_rules():
    # Surface domain
    d_surf = DomainClassifier.classify("Calculate CO adsorption on Pt(111) slab surface facet", "")
    assert d_surf == ScientificDomain.SURFACE_AND_INTERFACES

    # MD domain
    d_md = DomainClassifier.classify("Run molecular dynamics trajectory under NVT ensemble and calculate MSD diffusion", "")
    assert d_md == ScientificDomain.MOLECULAR_DYNAMICS

    # Structure domain
    d_struct = DomainClassifier.classify("Compute pair radial distribution function g(r) and neighbor list", "")
    assert d_struct == ScientificDomain.STRUCTURE_AND_COORDINATION

    # DFT domain
    d_dft = DomainClassifier.classify("Configure DFT kpoints and calculate band structure density of states", "")
    assert d_dft == ScientificDomain.DFT_AND_ELECTRONIC

    # Thermodynamics domain
    d_thermo = DomainClassifier.classify("Estimate free energy barrier and reaction coordinate residence time tau", "")
    assert d_thermo == ScientificDomain.THERMODYNAMICS_AND_KINETICS

    # Fallback general
    d_gen = DomainClassifier.classify("Analyze arbitrary numbers", "")
    assert d_gen == ScientificDomain.GENERAL_SIMULATION

    rules = DomainClassifier.get_rules_for_domain(ScientificDomain.SURFACE_AND_INTERFACES)
    assert len(rules) >= 4

def test_domain_rule_exception_resilience():
    # Ensure DomainRule gracefully returns False if a checker raises an exception
    def failing_checker(code, ast_info):
        raise ValueError("Simulated check failure")

    rule = DomainRule("test_fail", "Test failure rule", weight=10, checker=failing_checker)
    assert rule.evaluate("code", {}) is False

def test_dynamic_test_generator_empty():
    gen = DynamicTestGenerator()
    res = gen.generate_and_evaluate("")
    assert res["score"] == 0.0
    assert res["grade"] == "F"
    assert len(res["failed_checks"]) >= 1

def test_dynamic_test_generator_surface_workflow():
    prompt = "Write a Python script using ASE to calculate adsorption energy on a Cu(111) surface slab."
    script = """```python
import numpy as np
import ase
from ase.build import fcc111, molecule
import matplotlib.pyplot as plt

def build_slab_and_adsorb():
    slab = fcc111('Cu', size=(3, 3, 4), vacuum=10.0)
    co = molecule('CO')
    z_coords = slab.positions[:, 2]
    topmost_z = np.max(z_coords)
    return slab, co, topmost_z

def calculate_adsorption_energy(e_total, e_slab, e_co):
    e_ads = e_total - (e_slab + e_co)
    return e_ads

plt.figure()
plt.plot([1, 2], [-0.5, -0.6])
plt.savefig("adsorption_energy.png")
print("Adsorption calculation complete")
```"""
    gen = DynamicTestGenerator(prompt=prompt)
    res = gen.generate_and_evaluate(script)
    assert res["domain"] == ScientificDomain.SURFACE_AND_INTERFACES.value
    assert res["score"] >= 80.0
    assert res["grade"] in ("A", "B")
    assert len(res["passed_checks"]) >= 3
    assert any("test_syntax_compilation" in t["name"] for t in res["generated_tests"])

def test_dynamic_test_generator_md_workflow():
    prompt = "Run trajectory analysis to compute diffusion coefficient and MSD under NVT ensemble."
    script = """```python
import numpy as np
import ase
from ase.io import read
import matplotlib.pyplot as plt

def compute_msd(traj_file):
    traj = read(traj_file, index=':')
    displacements = []
    for frame in traj:
        displacements.append(np.mean(frame.positions**2))
    msd = np.array(displacements)
    diffusion = np.mean(msd) / (6.0 * 10.0)
    return msd, diffusion

plt.plot([0, 1, 2], [0, 1, 4])
plt.savefig("msd.png")
```"""
    gen = DynamicTestGenerator(prompt=prompt)
    res = gen.generate_and_evaluate(script)
    assert res["domain"] == ScientificDomain.MOLECULAR_DYNAMICS.value
    assert res["score"] >= 80.0
    assert res["grade"] in ("A", "B")

def test_dynamic_test_generator_dft_workflow():
    prompt = "Set up DFT kpoints and calculate total energy and forces."
    script = """```python
import numpy as np
import ase
from ase.calculators.vasp import Vasp

def setup_dft(atoms):
    calc = Vasp(kpts=[4, 4, 4], encut=400, xc='PBE')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    return energy, forces

with open("summary.txt", "w") as f:
    f.write("DFT converged")
```"""
    gen = DynamicTestGenerator(prompt=prompt)
    res = gen.generate_and_evaluate(script)
    assert res["domain"] == ScientificDomain.DFT_AND_ELECTRONIC.value
    assert res["score"] >= 70.0

def test_dynamic_test_generator_thermo_workflow():
    prompt = "Calculate free energy barrier and residence time tau."
    script = """```python
import numpy as np
import matplotlib.pyplot as plt

def compute_kinetics(time_lags):
    tau = 0.5
    fraction = 0.8
    p_t = np.exp(-time_lags / tau)
    return tau, fraction, p_t

plt.figure()
plt.plot([0, 1], [1, 0])
plt.savefig("rate.png")
```"""
    gen = DynamicTestGenerator(prompt=prompt)
    res = gen.generate_and_evaluate(script)
    assert res["domain"] == ScientificDomain.THERMODYNAMICS_AND_KINETICS.value
    assert res["score"] >= 70.0

def test_dynamic_test_generator_general_simulation():
    prompt = "Perform simulation analysis."
    script = """```python
import numpy as np
def run_simulation(x):
    data = np.linspace(0, 10, 100)
    result = np.mean(data)
    return result
print(run_simulation(5))
```"""
    gen = DynamicTestGenerator(prompt=prompt)
    res = gen.generate_and_evaluate(script)
    assert res["domain"] == ScientificDomain.GENERAL_SIMULATION.value
    assert res["score"] >= 60.0

def test_dynamic_test_generator_safety_audit_failure():
    prompt = "Calculate adsorption energy"
    unsafe_script = """```python
import os
os.system("del *.*")
```"""
    gen = DynamicTestGenerator(prompt=prompt)
    res = gen.generate_and_evaluate(unsafe_script)
    assert any("Security audit failed" in f for f in res["failed_checks"])

def test_benchmark_evaluator_direct():
    b_eval = BenchmarkEvaluator()
    assert "Water_Density_RDF_Survival_Probability" in b_eval.REFERENCE_NOTEBOOK
    empty_res = b_eval.evaluate("", "")
    assert empty_res["score"] == 0.0
    assert empty_res["grade"] == "F"

def test_hybrid_routing():
    evaluator = ScriptEvaluator()

    # 1. No prompt -> routes to flagship benchmark for backward compatibility
    res_default = evaluator.evaluate("")
    assert res_default["evaluation_type"] == "rag_benchmark"
    assert "reference_notebook" in res_default

    # 2. Flagship benchmark prompt -> routes to flagship benchmark
    flagship_prompt = "Run the interfacial water analysis benchmark on IrO2 rutile surface."
    res_flagship = evaluator.evaluate("", prompt=flagship_prompt)
    assert res_flagship["evaluation_type"] == "rag_benchmark"

    # 3. Out-of-RAG prompt -> routes to dynamic hybrid
    out_prompt = "Calculate Cu(111) surface adsorption energy of CO molecule using ASE."
    script = """```python
import numpy as np
import ase
from ase.build import fcc111
slab = fcc111('Cu', size=(2, 2, 3), vacuum=10.0)
z_vals = slab.positions[:, 2]
top_z = np.max(z_vals)
energy = -0.75
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, energy])
plt.savefig("out.png")
```"""
    res_dynamic = evaluator.evaluate(script, prompt=out_prompt)
    assert res_dynamic["evaluation_type"] == "dynamic_hybrid"
    assert res_dynamic["domain"] == ScientificDomain.SURFACE_AND_INTERFACES.value
    assert res_dynamic["score"] > 0
    assert "generated_tests" in res_dynamic
    assert res_dynamic["reference_notebook"] is None
    assert res_dynamic["metric_comparison"] is None

def test_dynamic_test_generator_syntax_error():
    gen = DynamicTestGenerator(prompt="Calculate surface energy")
    res = gen.generate_and_evaluate("def invalid_syntax(: pass")
    assert any("Syntax error in script" in f for f in res["failed_checks"])

def test_dynamic_test_generator_unvectorized_and_grades():
    # Loop-heavy unvectorized code
    prompt = "Process numerical data"
    script = """
for i in range(10):
    for j in range(10):
        for k in range(10):
            for m in range(10):
                for n in range(10):
                    pass
"""
    gen = DynamicTestGenerator(prompt=prompt)
    res = gen.generate_and_evaluate(script)
    assert any("Relies heavily on non-vectorized" in f for f in res["failed_checks"])
    assert res["grade"] in ("C", "D", "F")

def test_domain_rules_structure_and_coordination():
    prompt = "Compute RDF and coordination numbers"
    code = """
import numpy as np
import ase
from ase.neighborlist import neighbor_list
def calc_rdf(atoms):
    nl = neighbor_list('d', atoms, cutoff=3.0)
    g_r, bins = np.histogram(nl, bins=50)
    return g_r, bins
import matplotlib.pyplot as plt
plt.plot([1, 2], [3, 4])
plt.savefig("rdf.png")
"""
    gen = DynamicTestGenerator(prompt=prompt)
    res = gen.generate_and_evaluate(code)
    assert res["domain"] == ScientificDomain.STRUCTURE_AND_COORDINATION.value
    assert res["score"] >= 80.0

def test_script_evaluator_flagship_code_detection():
    evaluator = ScriptEvaluator()
    # Code contains signature benchmark variable m_sur
    code_with_sig = "m_sur = [1, 2]\no_bulk = [3, 4]\ncus_sites = [1]\nfull_analysis_presentation = 'fig.png'"
    assert evaluator.is_flagship_benchmark(prompt="Some generic prompt", code=code_with_sig) is True

def test_dynamic_test_generator_undefined_symbols_penalty():
    prompt = "Run surface calculation"
    code = "import numpy as np\nresult = undefined_func(missing_var)\n"
    gen = DynamicTestGenerator(prompt=prompt)
    res = gen.generate_and_evaluate(code)
    assert res["score"] <= 65.0
    assert any("Undefined variables or missing imports" in f for f in res["failed_checks"])

def test_dynamic_test_generator_runtime_crash_penalty():
    prompt = "Run calculation"
    code = "import numpy as np\nx = 1 / 0\n"
    gen = DynamicTestGenerator(prompt=prompt)
    res = gen.generate_and_evaluate(code)
    assert res["score"] <= 55.0
    assert res["grade"] in ("D", "F")
    assert any("Runtime execution failure" in f for f in res["failed_checks"])

def test_dynamic_test_generator_soft_plotting_and_fences():
    prompt = "General math plot"
    code = "```\nimport numpy as np\nimport matplotlib.pyplot as plt\nplt.title('Test')\n```"
    gen = DynamicTestGenerator(prompt=prompt)
    res = gen.generate_and_evaluate(code)
    assert res["score"] > 0
    assert any("Plotting calls detected" in p for p in res["passed_checks"])

def test_ast_analyzer_advanced_syntax():
    code = """
import numpy as np
from math import sqrt as my_sqrt

def my_func(a, *args, b=1, **kwargs):
    for x, y in [(1, 2)]:
        pass
    try:
        val = 10
    except ValueError as err:
        pass
    with open('test.txt') as handle:
        pass
    return a
"""
    analyzer = AstAnalyzer(code)
    analysis = analyzer.analyze()
    assert analysis["syntax_valid"] is True
    assert any(f["name"] == "my_func" for f in analysis["functions"])
    assert "undefined_symbols" in analysis

def test_dynamic_test_generator_timeout(monkeypatch):
    import subprocess
    prompt = "Timeout test"
    code = "import time\ntime.sleep(10)"
    gen = DynamicTestGenerator(prompt=prompt)

    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="python", timeout=5)

    monkeypatch.setattr(subprocess, "run", mock_run)
    res = gen.generate_and_evaluate(code)
    assert any("Execution timed out" in f for f in res["failed_checks"])

def test_dynamic_test_generator_grade_branches():
    gen = DynamicTestGenerator(prompt="Generic data analysis")
    # Score in A range (>=90)
    script_a = """
import numpy as np
import matplotlib.pyplot as plt

def process_data(vals):
    m = np.mean(vals)
    s = np.std(vals)
    return m, s

data = np.array([1.0, 2.0, 3.0])
mean_v, std_v = process_data(data)
plt.plot(data)
plt.savefig("plot.png")
print("Done")
"""
    res_a = gen.generate_and_evaluate(script_a)
    assert res_a["grade"] == "A"
    assert "summary" in res_a



