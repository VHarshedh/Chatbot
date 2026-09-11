import time
import json
import sys
import platform
from datetime import datetime

# Ensure stdout supports UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
        "id": "vasp_surface_poscar",
        "domain": "Surface Science / VASP",
        "title": "IrO₂(110) Surface Slab POSCAR",
        "prompt": (
            "Write a Python script using ASE and pymatgen to generate a VASP POSCAR file for a "
            "5-layer IrO2 rutile (110) surface slab with 15 Å vacuum and selective dynamics on the "
            "top 2 layers. The script must: (1) build the bulk IrO2 unit cell with correct lattice "
            "parameters (a=4.5051, c=3.1586 Å), (2) use SlabGenerator to cleave the (110) surface, "
            "(3) set selective dynamics flags, and (4) write the POSCAR with direct coordinates."
        )
    },
    {
        "id": "phonon_dispersion",
        "domain": "Phonon Calculations",
        "title": "Phonon Dispersion Setup",
        "prompt": (
            "Write a complete Python script using ASE and Phonopy to set up a phonon dispersion "
            "calculation for bulk silicon (diamond structure, a=5.431 Å). The script must: "
            "(1) create a 2x2x2 supercell, (2) generate displaced structures for force constants, "
            "(3) set up the band path along Γ-X-W-K-Γ-L high-symmetry points, and "
            "(4) plot the phonon band structure with correct axes labels and units (THz). "
            "Use finite displacement method with displacement distance 0.01 Å."
        )
    },
    {
        "id": "mlp_training_config",
        "domain": "ML Potentials",
        "title": "DeePMD Training JSON Configuration",
        "prompt": (
            "Write a complete DeePMD-kit v2.x input.json training configuration for a "
            "ternary Ir-O-H system using the se_e2_a descriptor. Include: "
            "(1) descriptor with rcut=6.0, rcut_smth=0.5, sel=[46,92,92], neuron=[25,50,100], "
            "(2) fitting_net with neuron=[240,240,240] and resnet_dt=true, "
            "(3) loss function with start_pref_e=0.02, limit_pref_e=1, start_pref_f=1000, limit_pref_f=1, "
            "(4) training with numb_steps=1000000, batch_size=auto, disp_freq=1000, save_freq=10000, "
            "(5) validation data path pointing to ./data/validation/. "
            "Output only the complete JSON block."
        )
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

# Timing threshold: warn if any individual benchmark exceeds this (seconds)
LATENCY_WARNING_THRESHOLD = 120.0


def run():
    agent = SingleShotAgent()
    evaluator = ScriptEvaluator()
    results = []

    run_timestamp = datetime.now().isoformat()
    total_benchmarks = len(BENCHMARKS)

    print("=" * 78)
    print(f"  BENCHMARKING {total_benchmarks} SCIENTIFIC PROMPTS")
    print(f"  Model: {agent.model}")
    print(f"  Timestamp: {run_timestamp}")
    print(f"  Platform: {platform.platform()} | Python {sys.version.split()[0]}")
    print("=" * 78)

    eval_scores = []
    warnings = []

    for i, bm in enumerate(BENCHMARKS, 1):
        print(f"\n{'─' * 78}")
        print(f"  [{i}/{total_benchmarks}] Domain: {bm['domain']}")
        print(f"  Title: {bm['title']}")
        print(f"  Prompt: {bm['prompt'][:90]}...")
        print(f"{'─' * 78}")

        t0 = time.time()
        res = agent.generate_response(bm["prompt"])
        duration = round(time.time() - t0, 2)
        status = res.get("status")

        response_text = res.get("response", "")
        thinking = res.get("thinking", "")
        content = res.get("content", "")
        eval_count = res.get("eval_count", 0)

        # Timing threshold check
        timing_flag = ""
        if duration > LATENCY_WARNING_THRESHOLD:
            timing_flag = " ⚠ SLOW"
            warnings.append(f"Benchmark '{bm['id']}' exceeded {LATENCY_WARNING_THRESHOLD}s ({duration}s)")

        print(f"  Status: {status} | Latency: {duration}s{timing_flag} | Tokens: {eval_count} | Thinking: {len(thinking.split())} words")

        eval_summary = None
        if status == "success":
            code = evaluator.extract_python_code(response_text)
            if code or bm["id"] == "interfacial_script_gen":
                eval_metrics = evaluator.evaluate(response_text, prompt=bm["prompt"])
                passed_cnt = len(eval_metrics.get("passed_checks", []))
                failed_cnt = len(eval_metrics.get("failed_checks", []))
                total_cnt = passed_cnt + failed_cnt
                eval_summary = {
                    "score": eval_metrics["score"],
                    "grade": eval_metrics["grade"],
                    "passed_count": passed_cnt,
                    "failed_count": failed_cnt,
                    "total_checks": total_cnt,
                    "summary": eval_metrics.get("summary", ""),
                    "evaluation_type": eval_metrics.get("evaluation_type", "benchmark")
                }
                eval_scores.append(eval_metrics["score"])
                grade_icon = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}.get(eval_metrics["grade"], "⚪")
                print(f"  {grade_icon} Score: {eval_metrics['score']}/100 | Grade: {eval_metrics['grade']} | Checks: {passed_cnt}/{total_cnt} | Mode: {eval_metrics.get('evaluation_type', 'eval')}")

        results.append({
            "id": bm["id"],
            "domain": bm["domain"],
            "title": bm["title"],
            "prompt": bm["prompt"],
            "status": status,
            "latency_seconds": duration,
            "latency_warning": duration > LATENCY_WARNING_THRESHOLD,
            "eval_count": eval_count,
            "thinking_preview": thinking[:200] + "..." if len(thinking) > 200 else thinking,
            "content_preview": content[:300] + "..." if len(content) > 300 else content,
            "evaluation": eval_summary
        })

    # ─── Summary Table ─────────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print("  BENCHMARK RESULTS SUMMARY")
    print("═" * 78)
    print(f"  {'#':<3} {'Domain':<28} {'Status':<10} {'Latency':>8}  {'Score':>7}  {'Grade':>5}")
    print(f"  {'─'*3} {'─'*28} {'─'*10} {'─'*8}  {'─'*7}  {'─'*5}")

    for i, r in enumerate(results, 1):
        status_str = "✓ PASS" if r["status"] == "success" else "✗ FAIL"
        latency_str = f"{r['latency_seconds']:>6.1f}s"
        if r["evaluation"]:
            score_str = f"{r['evaluation']['score']:>5.0f}/100"
            grade_str = f"  {r['evaluation']['grade']}"
        else:
            score_str = "    N/A"
            grade_str = "  N/A"
        warn = " ⚠" if r["latency_warning"] else ""
        print(f"  {i:<3} {r['domain']:<28} {status_str:<10} {latency_str}{warn}  {score_str}  {grade_str}")

    print(f"  {'─'*3} {'─'*28} {'─'*10} {'─'*8}  {'─'*7}  {'─'*5}")

    # Aggregate statistics
    total_passed = sum(1 for r in results if r["status"] == "success")
    total_failed = total_benchmarks - total_passed
    avg_latency = sum(r["latency_seconds"] for r in results) / total_benchmarks if total_benchmarks > 0 else 0
    avg_score = sum(eval_scores) / len(eval_scores) if eval_scores else 0

    print(f"\n  Passed: {total_passed}/{total_benchmarks} | Failed: {total_failed}/{total_benchmarks}")
    print(f"  Average Latency: {avg_latency:.1f}s")
    if eval_scores:
        print(f"  Average Evaluation Score: {avg_score:.1f}/100")

    if warnings:
        print(f"\n  ⚠ WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    • {w}")

    # ─── Export Results ──────────────────────────────────────────────────────────
    timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp_slug}.json"

    export_data = {
        "meta": {
            "timestamp": run_timestamp,
            "model": agent.model,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
            "total_benchmarks": total_benchmarks,
            "passed": total_passed,
            "failed": total_failed,
            "average_latency_seconds": round(avg_latency, 2),
            "average_eval_score": round(avg_score, 1) if eval_scores else None,
            "latency_warning_threshold_seconds": LATENCY_WARNING_THRESHOLD,
            "warnings": warnings
        },
        "results": results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    # Also write to the standard name for backward compatibility
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    print(f"\n  Results exported to: {output_file}")
    print(f"  Results exported to: benchmark_results.json")
    print("═" * 78)

if __name__ == "__main__":
    run()
