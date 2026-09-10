import re
import ast
from typing import Dict, Any, List, Tuple

class ScriptEvaluator:
    """
    Automated scientific evaluation rubric engine for zero-shot generated scripts
    on the Interfacial Water Dynamics & Surface Hydroxylation pipeline.
    Measures compliance against the 100-point ground truth standard.
    """

    @staticmethod
    def extract_python_code(raw_text: str) -> str:
        """Extracts Python code blocks from markdown-formatted LLM responses."""
        if not raw_text:
            return ""

        # Find fenced python code blocks
        blocks = re.findall(r"```(?:python)?\s*\n([\s\S]*?)```", raw_text, re.IGNORECASE)
        if blocks:
            # Return the largest code block
            return max(blocks, key=len).strip()

        # Fallback: if raw text already looks like python code
        lines = raw_text.strip().split("\n")
        code_lines = [l for l in lines if l.strip().startswith(("import ", "from ", "def ", "class ", "#", "for ", "if "))]
        if len(code_lines) >= 3:
            return raw_text.strip()

        return ""

    def evaluate(self, raw_response: str) -> Dict[str, Any]:
        """
        Evaluates a zero-shot generated script across 5 scientific categories (100 pts total).
        """
        code = self.extract_python_code(raw_response)
        if not code:
            return {
                "score": 0.0,
                "grade": "F",
                "extracted_code": "",
                "category_scores": {
                    "syntax_and_imports": {"score": 0, "max": 15},
                    "site_identification": {"score": 0, "max": 20},
                    "density_and_rdfs": {"score": 0, "max": 20},
                    "kinetics_and_survival": {"score": 0, "max": 25},
                    "model_averaging_and_outputs": {"score": 0, "max": 20}
                },
                "passed_checks": [],
                "failed_checks": ["No valid Python code found in the model response."],
                "summary": "Model failed to output a Python code block."
            }

        passed = []
        failed = []

        # =========================================================================
        # 1. Syntax & Core Imports (15 pts)
        # =========================================================================
        cat1_score = 0
        try:
            ast.parse(code)
            cat1_score += 5
            passed.append("Valid Python AST syntax (code compiles without syntax errors).")
        except SyntaxError as e:
            failed.append(f"Syntax error: {e.msg} at line {e.lineno}")

        # Import checks
        has_ase = bool(re.search(r"\b(import\s+ase|from\s+ase)\b", code))
        if has_ase:
            cat1_score += 4
            passed.append("Imported Atomic Simulation Environment (ASE) for trajectory handling.")
        else:
            failed.append("Missing ASE import for structure/trajectory loading.")

        has_numpy = bool(re.search(r"\b(import\s+numpy|from\s+numpy)\b", code))
        if has_numpy:
            cat1_score += 3
            passed.append("Imported NumPy for numerical operations and array manipulation.")
        else:
            failed.append("Missing NumPy import.")

        has_mpl = bool(re.search(r"\b(matplotlib|plt)\b", code))
        if has_mpl:
            cat1_score += 3
            passed.append("Imported Matplotlib for visualization and publication plots.")
        else:
            failed.append("Missing Matplotlib plotting imports.")

        # =========================================================================
        # 2. Interfacial Site Identification (20 pts)
        # =========================================================================
        cat2_score = 0

        # Surface metal detection
        has_msur = bool(re.search(r"(m_sur|surface_metal|topmost|max\s*\([^)]*z|z\s*>\s*|z_coord)", code, re.IGNORECASE))
        if has_msur:
            cat2_score += 5
            passed.append("Surface metal identification logic detected (identifies topmost metal layer).")
        else:
            failed.append("Missing surface metal layer identification logic.")

        # Bulk vs Water oxygen separation
        has_obulk = bool(re.search(r"(o_bulk|bulk_oxygen|o_w|water_oxygen|stoichiomet)", code, re.IGNORECASE))
        if has_obulk:
            cat2_score += 5
            passed.append("Bulk vs. water oxygen classification logic detected.")
        else:
            failed.append("Missing bulk vs water oxygen separation logic.")

        # CUS metal sites
        has_cus = bool(re.search(r"\b(cus|m_cus|coordinatively\s+unsaturated)\b", code, re.IGNORECASE))
        if has_cus:
            cat2_score += 5
            passed.append("Coordinatively Unsaturated Site (CUS) classification detected.")
        else:
            failed.append("Missing CUS site classification logic.")

        # Adsorbed & Bridge oxygens
        has_oadb_obr = bool(re.search(r"(o_adb|adsorbed|o_br|bridge|o_sur)", code, re.IGNORECASE))
        if has_oadb_obr:
            cat2_score += 5
            passed.append("Adsorbed oxygen (O_adb) and bridge/surface oxygen (O_br) classification detected.")
        else:
            failed.append("Missing adsorbed/bridge oxygen site tagging.")

        # =========================================================================
        # 3. Density Profiles & Pair RDFs (20 pts)
        # =========================================================================
        cat3_score = 0

        # z-density mass profile
        has_density = bool(re.search(r"(density|bin_width|histogram|bins_z|18\.015|mass_density)", code, re.IGNORECASE))
        if has_density:
            cat3_score += 10
            passed.append("Water mass density profile calculation along z-axis implemented.")
        else:
            failed.append("Missing z-density profile computation.")

        # Pair RDFs
        has_rdf = bool(re.search(r"\b(rdf|rpdf|pair_distribution|radial_distribution|g_r|dr\s*=)\b", code, re.IGNORECASE))
        if has_rdf:
            cat3_score += 10
            passed.append("Pair Radial Distribution Function (RDF) computation implemented.")
        else:
            failed.append("Missing pair radial distribution function (RDF) computation.")

        # =========================================================================
        # 4. Surface Kinetics & Survival Probability (25 pts)
        # =========================================================================
        cat4_score = 0

        # Hydroxyl / dissociation fraction
        has_hydroxyl = bool(re.search(r"(hydroxyl|dissociat|frac_oh|surv_oh|neighbor_list|cutoff.*1\.[1-5])", code, re.IGNORECASE))
        if has_hydroxyl:
            cat4_score += 12
            passed.append("Surface hydroxylation / water dissociation tracking implemented.")
        else:
            failed.append("Missing surface hydroxyl fraction / dissociation calculation.")

        # Survival probability P(t) with time lags
        has_survival = bool(re.search(r"(survival|p_t|residence|time_lag|lag|intersection|tau)", code, re.IGNORECASE))
        if has_survival:
            cat4_score += 13
            passed.append("Interfacial water survival probability P(t) tracking implemented.")
        else:
            failed.append("Missing water survival probability P(t) calculation.")

        # =========================================================================
        # 5. DP Model Averaging & Multi-Panel Artifacts (20 pts)
        # =========================================================================
        cat5_score = 0

        # Multi-model averaging across DP models
        has_multi_dp = bool(re.search(r"(dp_model|models|mean|std|average|cross_model|model_idx)", code, re.IGNORECASE))
        if has_multi_dp:
            cat5_score += 5
            passed.append("Multi-model trajectory loop / cross-model averaging (mean ± std) implemented.")
        else:
            failed.append("Missing multi-model DP loop or cross-model averaging logic.")

        # Individual figure saving
        has_indiv_plots = bool(re.search(r"(savefig|plot.*density|plot.*rdf|plot.*survival)", code, re.IGNORECASE))
        if has_indiv_plots:
            cat5_score += 5
            passed.append("Individual plot generation and file saving logic detected.")
        else:
            failed.append("Missing individual figure saving routines.")

        # Combined 2x2 multi-panel summary figure
        has_2x2 = bool(re.search(r"(subplots\s*\(\s*2\s*,\s*2|2x2|2,\s*2\s*summary|grid|4.panel)", code, re.IGNORECASE))
        if has_2x2:
            cat5_score += 5
            passed.append("Combined 2x2 multi-panel summary figure layout implemented.")
        else:
            failed.append("Missing combined 2x2 summary figure creation (e.g. plt.subplots(2, 2)).")

        # Text summary report
        has_report = bool(re.search(r"(summary_report\.txt|report|residence_time|txt.*write)", code, re.IGNORECASE))
        if has_report:
            cat5_score += 5
            passed.append("Text summary report generation (summary_report.txt) implemented.")
        else:
            failed.append("Missing summary_report.txt output generation.")

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
            f"Zero-shot script achieved an accuracy score of {total_score}/100 (Grade {grade}). "
            f"Passed {len(passed)}/16 ground truth checks. "
        )
        if total_score >= 80:
            summary += "The model demonstrated strong zero-shot scientific comprehension of interfacial water workflows."
        elif total_score >= 60:
            summary += "The model grasped the general pipeline structure but missed key mathematical or site classification details."
        else:
            summary += "The model struggled with zero-shot generation of the complete multi-step scientific analysis pipeline."

        return {
            "score": total_score,
            "grade": grade,
            "extracted_code": code,
            "category_scores": {
                "syntax_and_imports": {"score": cat1_score, "max": 15, "pct": round(cat1_score / 15 * 100)},
                "site_identification": {"score": cat2_score, "max": 20, "pct": round(cat2_score / 20 * 100)},
                "density_and_rdfs": {"score": cat3_score, "max": 20, "pct": round(cat3_score / 20 * 100)},
                "kinetics_and_survival": {"score": cat4_score, "max": 25, "pct": round(cat4_score / 25 * 100)},
                "model_averaging_and_outputs": {"score": cat5_score, "max": 20, "pct": round(cat5_score / 20 * 100)}
            },
            "passed_checks": passed,
            "failed_checks": failed,
            "summary": summary
        }
