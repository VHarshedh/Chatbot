import re
import ast
from typing import Dict, Any, List, Tuple, Optional

class ScriptEvaluator:
    """
    Automated scientific evaluation rubric engine for generated analysis scripts
    benchmarked against the Interfacial Water Dynamics & Surface Hydroxylation pipeline
    defined in:
    ChatBot/Analysis/Water_Density_RDF_Survival_Probability_Hydroxyl_Fraction_with_IrO2_Template_Seperate_Plot_Capability.ipynb
    
    Measures compliance against the 100-point ground truth standard across 5 categories,
    verifies notebook-specific algorithmic features, and provides numerical metric comparison
    with reference ground-truth outputs within a ±5% relative tolerance.
    """

    REFERENCE_NOTEBOOK: str = (
        "ChatBot/Analysis/Water_Density_RDF_Survival_Probability_Hydroxyl_Fraction_with_"
        "IrO2_Template_Seperate_Plot_Capability.ipynb"
    )

    # Reference metrics extracted directly from the notebook cell outputs
    REFERENCE_METRICS: Dict[str, Dict[str, Any]] = {
        "tau_oh_ns": {
            "value": 0.298,
            "std": 0.114,
            "unit": "ns",
            "name": "Residence Time tau(OD/OH)",
            "description": "Continuous residence time of surface hydroxyl state (OD/OH)"
        },
        "tau_h2o_ns": {
            "value": 0.043,
            "std": 0.028,
            "unit": "ns",
            "name": "Residence Time tau(D2O/H2O)",
            "description": "Continuous residence time of intact molecular water state (D2O/H2O)"
        },
        "frac_oh_pct": {
            "value": 77.44,
            "std": 6.48,
            "unit": "%",
            "name": "Dissociated Fraction (OD/OH)",
            "description": "Average equilibrium surface dissociation percentage"
        },
        "frac_h2o_pct": {
            "value": 22.56,
            "std": 6.48,
            "unit": "%",
            "name": "Molecular Fraction (D2O/H2O)",
            "description": "Average equilibrium surface molecular water percentage"
        }
    }

    @staticmethod
    def extract_python_code(raw_text: str) -> str:
        """Extracts Python code blocks from markdown-formatted LLM responses."""
        if not raw_text:
            return ""

        # 1. Strip out thinking details blocks so HTML details never contaminate Python code
        cleaned_text = re.sub(r"<details[\s\S]*?<\/details>", "", raw_text, flags=re.IGNORECASE).strip()

        # 2. Match fenced python code blocks with or without closing fence
        blocks = re.findall(r"```(?:python|py)?\s*?\n?([\s\S]*?)(?:```|$)", cleaned_text, re.IGNORECASE)
        
        # Filter for candidates that actually contain code statements
        code_candidates = []
        for b in blocks:
            code_candidate = b.strip()
            if not code_candidate:
                continue
            lines = [l.strip() for l in code_candidate.splitlines() if l.strip() and not l.strip().startswith("#")]
            if any(l.startswith(("import ", "from ", "def ", "class ", "for ", "while ", "with ", "atoms", "m =")) or "=" in l for l in lines):
                code_candidates.append(code_candidate)

        if code_candidates:
            return max(code_candidates, key=len).strip()

        # 3. Fallback: find the first import or def statement and extract down
        lines = cleaned_text.splitlines()
        start_idx = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ", "def ", "class ")) and not stripped.startswith(("<", "#")):
                start_idx = i
                break

        if start_idx != -1:
            extracted = "\n".join(lines[start_idx:]).strip()
            extracted = re.sub(r"```+$", "", extracted).strip()
            return extracted

        return ""

    @staticmethod
    def extract_metrics(text: str) -> Dict[str, float]:
        """
        Parses numerical metrics (tau_oh, tau_h2o, frac_oh, frac_h2o) from generated
        script comments, print outputs, or summary reports.
        """
        if not text:
            return {}
        metrics: Dict[str, float] = {}

        # tau_oh / tau_od (ns)
        m_tau_oh = re.search(r"(?:tau[_\s]*(?:oh|od)|residence\s*time[_\s]*(?:oh|od))[^:\n]*[:\s=]+([0-9.]+)", text, re.IGNORECASE)
        if m_tau_oh:
            try:
                metrics["tau_oh_ns"] = float(m_tau_oh.group(1))
            except ValueError:
                pass

        # tau_h2o / tau_d2o (ns)
        m_tau_h2o = re.search(r"(?:tau[_\s]*(?:h2o|d2o)|residence\s*time[_\s]*(?:h2o|d2o))[^:\n]*[:\s=]+([0-9.]+)", text, re.IGNORECASE)
        if m_tau_h2o:
            try:
                metrics["tau_h2o_ns"] = float(m_tau_h2o.group(1))
            except ValueError:
                pass

        # frac_oh / dissociated fraction (%)
        m_frac_oh = re.search(
            r"(?:dissociat[a-z]*|frac[_\s]*oh|surface\s*dissociation)[^:\n]*[:\s=]+([0-9.]+)\s*%?",
            text,
            re.IGNORECASE
        )
        if m_frac_oh:
            try:
                metrics["frac_oh_pct"] = float(m_frac_oh.group(1))
            except ValueError:
                pass

        # frac_h2o / molecular fraction (%)
        m_frac_h2o = re.search(
            r"(?:molecular|intact|frac[_\s]*h2o)[^:\n]*[:\s=]+([0-9.]+)\s*%?",
            text,
            re.IGNORECASE
        )
        if m_frac_h2o:
            try:
                metrics["frac_h2o_pct"] = float(m_frac_h2o.group(1))
            except ValueError:
                pass

        return metrics

    def compare_metrics(
        self,
        candidate_metrics: Dict[str, float],
        tolerance: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compares candidate numerical metrics against the notebook's ground-truth values
        within the requested tolerance (default ±5% relative error).
        """
        comparison: Dict[str, Any] = {}
        passed_count = 0
        total_compared = 0

        for key, ref in self.REFERENCE_METRICS.items():
            if key in candidate_metrics:
                c_val = candidate_metrics[key]
                r_val = ref["value"]
                diff = abs(c_val - r_val)
                rel_err = diff / r_val if r_val != 0 else 0.0
                is_pass = rel_err <= tolerance
                if is_pass:
                    passed_count += 1
                total_compared += 1

                comparison[key] = {
                    "name": ref["name"],
                    "unit": ref["unit"],
                    "candidate_value": c_val,
                    "reference_value": r_val,
                    "reference_std": ref["std"],
                    "relative_error_pct": round(rel_err * 100, 2),
                    "tolerance_pct": round(tolerance * 100, 1),
                    "passed": is_pass
                }

        accuracy_score = round((passed_count / total_compared * 100), 1) if total_compared > 0 else 0.0

        return {
            "tolerance": tolerance,
            "metrics": comparison,
            "passed_count": passed_count,
            "total_compared": total_compared,
            "all_passed": (passed_count == total_compared) if total_compared > 0 else False,
            "metrics_accuracy_score": accuracy_score
        }

    def inspect_notebook_features(self, code: str) -> Dict[str, Any]:
        """
        Detects specific algorithmic implementation features mirroring the reference notebook:
        Water_Density_RDF_Survival_Probability_Hydroxyl_Fraction_with_IrO2_Template_Seperate_Plot_Capability.ipynb
        """
        if not code:
            return {}

        return {
            # 1. ASE neighbor list API (Notebook Cell 00 & Cell 05)
            "ase_neighborlist_api": bool(
                re.search(r"(neighbor_list|NeighborList|natural_cutoffs)", code)
            ),
            # 2. Stoichiometry bulk oxide separation s * N_metal (Notebook Cell 02 line 59)
            "stoichiometry_bulk_filter": bool(
                re.search(r"(s\s*\*\s*len|s\s*\*\s*N_metal|o_bul_vals\s*=\s*o_z\[\s*:\s*\(\s*s)", code)
            ),
            # 3. CUS distance cutoff 2.5 Angstroms (Notebook Cell 02 lines 82-83, Cell 05 line 63)
            "cus_distance_cutoff_2_5A": bool(
                re.search(r"(get_distance\s*\([^)]*2\.5|dist\s*<=\s*2\.5|cutoff\s*=\s*2\.5)", code)
            ),
            # 4. O-H bonding cutoff 1.2 Angstroms (Notebook Cell 05 line 75)
            "oh_coordination_cutoff_1_2A": bool(
                re.search(r"(\('O',\s*'H'\)\s*:\s*1\.[1-3]|cutoff\s*=\s*1\.[1-3]|1\.2)", code)
            ),
            # 5. Mass density conversion formula with 18.015 g/mol (Notebook Cell 02 lines 124-125)
            "mass_density_conversion_18_015": bool(
                re.search(r"(18\.015|6\.023e23|volume\s*\*\s*1e-24)", code)
            ),
            # 6. Pair RDFs (Notebook Cell 02 lines 139-144)
            "pair_rdfs_present": bool(
                re.search(r"(rpdf|atoms2struct|g_r_mow|g_r_osh|g_r_osow|dr\s*=\s*0\.05|pair_distribution)", code)
            ),
            # 7. Continuous survival probability via set intersection (Notebook Cell 05 line 105)
            "continuous_survival_intersection": bool(
                re.search(r"(set\.intersection|intersection\s*\(|oh_sets)", code)
            ),
            # 8. Numerical integration for residence time tau (Notebook Cell 06 lines 17-18)
            "trapezoidal_tau_integration": bool(
                re.search(r"(np\.trapezoid|np\.trapz|trapezoid\s*\(|trapz\s*\()", code)
            ),
            # 9. Intermediate NPZ results persistence (Notebook Cell 02 line 162, Cell 05 line 117)
            "npz_persistence": bool(
                re.search(r"(simulation_results\.npz|survival_fraction_results\.npz|np\.savez)", code)
            ),
            # 10. Combined 2x2 presentation layout (Notebook Cell 07 line 25)
            "consolidated_2x2_presentation": bool(
                re.search(r"(subplots\s*\(\s*2\s*,\s*2|full_analysis_presentation\.png|2x2)", code)
            ),
            # 11. Summary report output (Notebook Cell 07 line 115)
            "summary_report_txt": bool(
                re.search(r"(summary_report\.txt|report)", code)
            )
        }

    def evaluate(self, raw_response: str) -> Dict[str, Any]:
        """
        Evaluates a zero-shot generated script across 5 scientific categories (100 pts total)
        benchmarked against the reference analysis notebook:
        Water_Density_RDF_Survival_Probability_Hydroxyl_Fraction_with_IrO2_Template_Seperate_Plot_Capability.ipynb
        """
        code = self.extract_python_code(raw_response)
        if not code:
            return {
                "score": 0.0,
                "grade": "F",
                "extracted_code": "",
                "reference_notebook": self.REFERENCE_NOTEBOOK,
                "category_scores": {
                    "syntax_and_imports": {"score": 0, "max": 15, "pct": 0},
                    "site_identification": {"score": 0, "max": 20, "pct": 0},
                    "density_and_rdfs": {"score": 0, "max": 20, "pct": 0},
                    "kinetics_and_survival": {"score": 0, "max": 25, "pct": 0},
                    "model_averaging_and_outputs": {"score": 0, "max": 20, "pct": 0}
                },
                "passed_checks": [],
                "failed_checks": ["No valid Python code found in the model response."],
                "notebook_features": {},
                "metric_comparison": None,
                "summary": "Model failed to output a Python code block."
            }

        passed: List[str] = []
        failed: List[str] = []

        # =========================================================================
        # 1. Syntax & Core Imports (15 pts) [Notebook Cell 00]
        # =========================================================================
        cat1_score = 0
        try:
            ast.parse(code)
            cat1_score += 5
            passed.append("Valid Python AST syntax (code compiles without syntax errors).")
        except SyntaxError as e:
            failed.append(f"Syntax error: {e.msg} at line {e.lineno}")

        # ASE imports (Notebook Cell 00: ase, io, Atoms, neighbor_list)
        has_ase = bool(re.search(r"\b(import\s+ase|from\s+ase)\b", code))
        if has_ase:
            cat1_score += 4
            passed.append("Imported Atomic Simulation Environment (ASE) for trajectory handling (Notebook Cell 00: io, Atoms, neighbor_list).")
        else:
            failed.append("Missing ASE import for structure/trajectory loading (Notebook Cell 00).")

        # NumPy import (Notebook Cell 00)
        has_numpy = bool(re.search(r"\b(import\s+numpy|from\s+numpy)\b", code))
        if has_numpy:
            cat1_score += 3
            passed.append("Imported NumPy for numerical operations and array manipulation (Notebook Cell 00).")
        else:
            failed.append("Missing NumPy import (Notebook Cell 00).")

        # Matplotlib visualization imports (Notebook Cell 00/03/07)
        has_mpl = bool(re.search(r"\b(matplotlib|plt|pylab)\b", code))
        if has_mpl:
            cat1_score += 3
            passed.append("Imported Matplotlib for visualization and publication plots (Notebook Cell 00/03/07).")
        else:
            failed.append("Missing Matplotlib plotting imports.")

        # =========================================================================
        # 2. Interfacial Site Identification (20 pts) [Notebook Cell 02 & Cell 05]
        # =========================================================================
        cat2_score = 0

        # Surface metal detection (Notebook Cell 02 line 56, Cell 05 line 52: max(m_z) - 1.5)
        has_msur = bool(re.search(r"(m_sur|surface_metal|topmost|max\s*\([^)]*z|z\s*>\s*|z_coord|m_z)", code, re.IGNORECASE))
        if has_msur:
            cat2_score += 5
            passed.append("Surface metal identification logic detected (identifies topmost metal layer, Notebook Cell 02 line 56, Cell 05 line 52).")
        else:
            failed.append("Missing surface metal layer identification logic (Notebook Cell 02 line 56).")

        # Bulk vs Water oxygen separation via stoichiometry (Notebook Cell 02 lines 58-75: s * len(m_z))
        has_obulk = bool(re.search(r"(o_bul|bulk_oxygen|o_w|water_oxygen|stoichiomet|s\s*\*\s*len|s\s*\*\s*N_metal)", code, re.IGNORECASE))
        if has_obulk:
            cat2_score += 5
            passed.append("Bulk vs. water oxygen classification logic detected via oxide stoichiometry (Notebook Cell 02 lines 58-75).")
        else:
            failed.append("Missing bulk vs water oxygen separation logic (Notebook Cell 02 lines 58-75).")

        # CUS metal sites (Notebook Cell 02 line 78)
        has_cus = bool(re.search(r"\b(cus|m_cus|coordinatively\s+unsaturated)\b", code, re.IGNORECASE))
        if has_cus:
            cat2_score += 5
            passed.append("Coordinatively Unsaturated Site (CUS) classification detected (Notebook Cell 02 line 78).")
        else:
            failed.append("Missing CUS site classification logic (Notebook Cell 02 line 78).")

        # Adsorbed & Bridge oxygens via 2.5A cutoff (Notebook Cell 02 lines 79-94, Cell 05 lines 58-66)
        has_oadb_obr = bool(re.search(r"(o_adb|adsorbed|o_br|bridge|o_sur)", code, re.IGNORECASE))
        if has_oadb_obr:
            cat2_score += 5
            passed.append("Adsorbed oxygen (O_adb) and bridge/surface oxygen (O_br) classification detected via 2.5 Å cutoff (Notebook Cell 02 lines 79-94).")
        else:
            failed.append("Missing adsorbed/bridge oxygen site tagging (Notebook Cell 02 lines 79-94).")

        # =========================================================================
        # 3. Density Profiles & Pair RDFs (20 pts) [Notebook Cell 02 & Cell 03]
        # =========================================================================
        cat3_score = 0

        # z-density mass profile (Notebook Cell 02 lines 95-129)
        has_density = bool(re.search(r"(density|bin_width|histogram|bins_z|18\.015|mass_density)", code, re.IGNORECASE))
        if has_density:
            cat3_score += 10
            passed.append("Water mass density profile calculation along z-axis implemented (Notebook Cell 02 lines 95-129).")
        else:
            failed.append("Missing z-density profile computation (Notebook Cell 02 lines 95-129).")

        # Pair RDFs (Notebook Cell 02 lines 130-148: M_cus-Ow, Obr-Hw, Obr-Ow)
        has_rdf = bool(re.search(r"\b(rdf|rpdf|pair_distribution|radial_distribution|g_r|dr\s*=)\b", code, re.IGNORECASE))
        if has_rdf:
            cat3_score += 10
            passed.append("Pair Radial Distribution Function (RDF) computation implemented (Notebook Cell 02 lines 130-148).")
        else:
            failed.append("Missing pair radial distribution function (RDF) computation (Notebook Cell 02 lines 130-148).")

        # =========================================================================
        # 4. Surface Kinetics & Survival Probability (25 pts) [Notebook Cell 05 & Cell 06]
        # =========================================================================
        cat4_score = 0

        # Hydroxyl / dissociation fraction with 1.2A cutoff (Notebook Cell 05 lines 67-88)
        has_hydroxyl = bool(re.search(r"(hydroxyl|dissociat|frac_oh|surv_oh|neighbor_list|cutoff.*1\.[1-5]|indx_oh)", code, re.IGNORECASE))
        if has_hydroxyl:
            cat4_score += 12
            passed.append("Surface hydroxylation / water dissociation tracking implemented with 1.2 Å neighbor cutoff (Notebook Cell 05 lines 67-88).")
        else:
            failed.append("Missing surface hydroxyl fraction / dissociation calculation (Notebook Cell 05 lines 67-88).")

        # Survival probability P(t) with time lags and residence time tau (Notebook Cell 05 lines 89-110, Cell 06 lines 14-28)
        has_survival = bool(re.search(r"(survival|p_t|residence|time_lag|lag|intersection|tau)", code, re.IGNORECASE))
        if has_survival:
            cat4_score += 13
            passed.append("Interfacial water survival probability P(t) tracking implemented via set intersection and residence time integration (Notebook Cell 05 lines 89-110, Cell 06).")
        else:
            failed.append("Missing water survival probability P(t) calculation (Notebook Cell 05 lines 89-110).")

        # =========================================================================
        # 5. DP Model Averaging & Multi-Panel Artifacts (20 pts) [Notebook Cell 02/05/07]
        # =========================================================================
        cat5_score = 0

        # Multi-model averaging across DP models (Notebook Cell 02 lines 102-155, Cell 05 line 112)
        has_multi_dp = bool(re.search(r"(dp_model|models|mean|std|average|cross_model|model_idx|modelz|model_num)", code, re.IGNORECASE))
        if has_multi_dp:
            cat5_score += 5
            passed.append("Multi-model trajectory loop / cross-model averaging (mean ± std) implemented across DP models (Notebook Cell 02 line 102, Cell 05 line 20).")
        else:
            failed.append("Missing multi-model DP loop or cross-model averaging logic.")

        # Individual figure saving / plot routines (Notebook Cell 03)
        has_indiv_plots = bool(re.search(r"(savefig|plot.*density|plot.*rdf|plot.*survival|fig1|fig2|plt\.show)", code, re.IGNORECASE))
        if has_indiv_plots:
            cat5_score += 5
            passed.append("Individual plot generation and file saving logic detected (Notebook Cell 03).")
        else:
            failed.append("Missing individual figure saving routines.")

        # Combined 2x2 multi-panel summary figure (Notebook Cell 07 lines 25-95: full_analysis_presentation.png)
        has_2x2 = bool(re.search(r"(subplots\s*\(\s*2\s*,\s*2|2x2|2,\s*2|grid|4.panel|axs\s*=\s*axs|full_analysis_presentation)", code, re.IGNORECASE))
        if has_2x2:
            cat5_score += 5
            passed.append("Combined 2x2 multi-panel summary figure layout implemented (Notebook Cell 07 lines 25-95).")
        else:
            failed.append("Missing combined 2x2 summary figure creation (e.g. plt.subplots(2, 2)).")

        # Text summary report (Notebook Cell 07 lines 115-125: summary_report.txt)
        has_report = bool(re.search(r"(summary_report\.txt|report|residence_time|txt.*write|summary_report)", code, re.IGNORECASE))
        if has_report:
            cat5_score += 5
            passed.append("Text summary report generation (summary_report.txt) implemented (Notebook Cell 07 lines 115-125).")
        else:
            failed.append("Missing summary_report.txt output generation (Notebook Cell 07 line 115).")

        # =========================================================================
        # Total Score & Grade Calculation
        # =========================================================================
        total_score = round(cat1_score + cat2_score + cat3_score + cat4_score + cat5_score, 1)

        if total_score >= 90:
            grade = "A"
        elif total_score >= 80:
            grade = "B"
        elif total_score >= 70:
            grade = "C"
        elif total_score >= 60:
            grade = "D"
        else:
            grade = "F"

        summary = (
            f"Zero-shot script achieved an accuracy score of {total_score}/100 (Grade {grade}) "
            f"against the reference notebook standard. Passed {len(passed)}/16 scientific checks. "
        )
        if total_score >= 90:
            summary += "Outstanding compliance: The generated code faithfully replicates all stages of the reference analysis pipeline."
        elif total_score >= 80:
            summary += "The model demonstrated strong scientific comprehension of interfacial water workflows from the notebook."
        elif total_score >= 60:
            summary += "The model grasped the general pipeline structure but missed key mathematical or site classification details."
        else:
            summary += "The model struggled with zero-shot generation of the complete multi-step scientific analysis pipeline."

        # Notebook specific feature inspection
        notebook_features = self.inspect_notebook_features(code)

        # Numerical metric comparison (if metrics exist in the text/code)
        extracted_cand = self.extract_metrics(raw_response)
        metric_comparison = self.compare_metrics(extracted_cand, tolerance=0.05) if extracted_cand else None

        return {
            "score": total_score,
            "grade": grade,
            "extracted_code": code,
            "reference_notebook": self.REFERENCE_NOTEBOOK,
            "category_scores": {
                "syntax_and_imports": {"score": cat1_score, "max": 15, "pct": round(cat1_score / 15 * 100)},
                "site_identification": {"score": cat2_score, "max": 20, "pct": round(cat2_score / 20 * 100)},
                "density_and_rdfs": {"score": cat3_score, "max": 20, "pct": round(cat3_score / 20 * 100)},
                "kinetics_and_survival": {"score": cat4_score, "max": 25, "pct": round(cat4_score / 25 * 100)},
                "model_averaging_and_outputs": {"score": cat5_score, "max": 20, "pct": round(cat5_score / 20 * 100)}
            },
            "passed_checks": passed,
            "failed_checks": failed,
            "notebook_features": notebook_features,
            "metric_comparison": metric_comparison,
            "summary": summary
        }
