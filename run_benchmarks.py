import time
import json
from backend.agents.baseline_agent import SingleShotAgent
from backend.evaluation.script_evaluator import ScriptEvaluator

BENCHMARKS = [
    {
        "id": "dft_cutoff",
        "domain": "DFT",
        "title": "Pt SCF Convergence Parameters",
        "prompt": "What are the recommended plane-wave kinetic energy cutoff (ecutwfc) and charge density cutoff (ecutrho) for Platinum (Pt) using a PBE PAW or ONCV pseudopotential in Quantum ESPRESSO? Please write a sample &SYSTEM namelist block."
    },
    {
        "id": "md_chaining",
        "domain": "Molecular Dynamics",
        "title": "DeePMD LAMMPS Interface Script",
        "prompt": "Write a minimal LAMMPS in.lammps input script for an NVT production run of an interfacial water system driven by a pair_style deepmd potential. Include velocity creation, thermostatting at 300K, and periodic restart writes."
    },
    {
        "id": "training_format",
        "domain": "DeepMD Training",
        "title": "Raw Data to Binary Sets",
        "prompt": "In DeePMD-kit, what is the exact layout of raw data files (coord.raw, box.raw, energy.raw, force.raw) and how are they converted into set.000 numpy binary datasets for training?"
    },
    {
        "id": "analysis_kinetics",
        "domain": "Interfacial Analysis",
        "title": "Surface Sites & Survival Probability",
        "prompt": "Explain how coordinatively unsaturated (CUS) metal sites and adsorbed water oxygens (O_adb) are structurally classified on a metal oxide surface, and write the mathematical definition of the water survival probability P(t)."
    },
    {
        "id": "interfacial_script_gen",
        "domain": "Interfacial Script Evaluation",
        "title": "Full Interfacial Analysis Script Generator",
        "prompt": (
            "Write a complete, self-contained scientific Python script using ASE, NumPy, and Matplotlib to analyze "
            "an interfacial metal-oxide/water system (template: IrO2/H2O) from DeepMD .extxyz trajectories across multiple DP models. "
            "Your script must:\n"
            "1. Classify surface metal atoms (M_sur from topmost z-layer), bulk vs water oxygens (O_bulk vs O_w via stoichiometry), "
            "coordinatively unsaturated metal sites (CUS), and adsorbed vs bridge oxygens (O_adb, O_br).\n"
            "2. Compute water mass density profiles along the z-axis (g/cm^3) and pair radial distribution functions (RDFs: M_cus-O_w, O_br-H_w, O_br-O_w).\n"
            "3. Track surface hydroxylation / dissociation fractions (OH vs intact H2O) using neighbor distance cutoffs (~1.2 A), and compute the water survival probability P(t) over log-spaced time lags tau via continuous set intersections.\n"
            "4. Loop over multiple DP models, computing the cross-model mean and standard deviation (mean +/- std).\n"
            "5. Save individual plots for density, RDF, survival, and fraction, as well as a combined 2x2 multi-panel summary presentation figure (full_analysis_presentation.png).\n"
            "6. Write a summary_report.txt file with average residence times (integrated from P(t)) and dissociated/molecular fractions.\n"
            "Output only the complete Python code block."
        )
    }
]

def run():
    agent = SingleShotAgent()
    evaluator = ScriptEvaluator()
    results = []

    print("=" * 70)
    print(f"BENCHMARKING 5 SCIENTIFIC PROMPTS WITH MODEL: {agent.model}")
    print("=" * 70)

    for i, bm in enumerate(BENCHMARKS, 1):
        print(f"\n[{i}/5] Testing Domain: {bm['domain']} | Title: {bm['title']}")
        print(f"Prompt preview: {bm['prompt'][:80]}...")
        t0 = time.time()

        res = agent.generate_response(bm["prompt"])
        duration = round(time.time() - t0, 2)
        status = res.get("status")

        response_text = res.get("response", "")
        thinking = res.get("thinking", "")
        content = res.get("content", "")
        eval_count = res.get("eval_count", 0)

        print(f"Status: {status} | Latency: {duration}s | Tokens: {eval_count} | Thinking Words: {len(thinking.split())}")

        eval_summary = None
        if bm["id"] == "interfacial_script_gen" and status == "success":
            eval_metrics = evaluator.evaluate(response_text)
            eval_summary = {
                "score": eval_metrics["score"],
                "grade": eval_metrics["grade"],
                "passed_count": len(eval_metrics["passed_checks"]),
                "failed_count": len(eval_metrics["failed_checks"]),
                "summary": eval_metrics["summary"]
            }
            print(f"Accuracy Evaluation: Score {eval_metrics['score']}/100 | Grade: {eval_metrics['grade']} | Passed Checks: {len(eval_metrics['passed_checks'])}/16")

        results.append({
            "id": bm["id"],
            "domain": bm["domain"],
            "title": bm["title"],
            "prompt": bm["prompt"],
            "status": status,
            "latency_seconds": duration,
            "eval_count": eval_count,
            "thinking_preview": thinking[:200] + "..." if len(thinking) > 200 else thinking,
            "content_preview": content[:300] + "..." if len(content) > 300 else content,
            "evaluation": eval_summary
        })

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL 5 BENCHMARKS COMPLETED! Results saved to benchmark_results.json")
    print("=" * 70)

if __name__ == "__main__":
    run()
