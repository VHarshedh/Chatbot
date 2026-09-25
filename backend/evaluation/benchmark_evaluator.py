import re
import ast
from typing import Dict, Any, List, Optional

class BenchmarkEvaluator:
    """
    Standard ground-truth rubric engine for the flagship Interfacial Water Dynamics benchmark
    defined in:
    ChatBot/Analysis/Water_Density_RDF_Survival_Probability_Hydroxyl_Fraction_with_IrO2_Template_Seperate_Plot_Capability.ipynb
    
    Strict 100-point rubric with fine-grained sub-checks, tightened tolerances,
    and severity-tiered grade caps.
    """

    REFERENCE_NOTEBOOK: str = (
        "ChatBot/Analysis/Water_Density_RDF_Survival_Probability_Hydroxyl_Fraction_with_"
        "IrO2_Template_Seperate_Plot_Capability.ipynb"
    )

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

    # Tightened from 5% to 3% for stricter metric comparison
    DEFAULT_METRIC_TOLERANCE: float = 0.03

    @staticmethod
    def extract_metrics(text: str) -> Dict[str, float]:
        """Parses numerical metrics from script comments, print outputs, or summary reports."""
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
        tolerance: float = None
    ) -> Dict[str, Any]:
        """Compares candidate numerical metrics against reference ground-truth values."""
        if tolerance is None:
            tolerance = self.DEFAULT_METRIC_TOLERANCE

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
        """Detects specific algorithmic implementation features from the notebook."""
        if not code:
            return {}

        return {
            "ase_neighborlist_api": bool(
                re.search(r"(neighbor_list|NeighborList|natural_cutoffs)", code)
            ),
            "ase_io_read": bool(
                re.search(r"(from\s+ase\.io\s+import|ase\.io\.read)", code)
            ),
            "stoichiometry_bulk_filter": bool(
                re.search(r"(s\s*\*\s*len|s\s*\*\s*N_metal|o_bul_vals\s*=\s*o_z\[\s*:\s*\(\s*s)", code)
            ),
            "cus_distance_cutoff_2_5A": bool(
                re.search(r"(get_distance\s*\([^)]*2\.5|dist\s*<=\s*2\.5|cutoff\s*=\s*2\.5)", code)
            ),
            "oh_coordination_cutoff_1_2A": bool(
                re.search(r"(\('O',\s*'H'\)\s*:\s*1\.[1-3]|cutoff\s*=\s*1\.[1-3]|1\.2)", code)
            ),
            "mass_density_conversion_18_015": bool(
                re.search(r"(18\.015|6\.022e23|6\.023e23|volume\s*\*\s*1e-24)", code)
            ),
            "pair_rdfs_present": bool(
                re.search(r"(rpdf|atoms2struct|g_r_mow|g_r_osh|g_r_osow|dr\s*=\s*0\.05|pair_distribution)", code)
            ),
            "log_spaced_lags": bool(
                re.search(r"(logspace|geomspace|np\.log)", code)
            ),
            "continuous_survival_intersection": bool(
                re.search(r"(set\.intersection|intersection\s*\(|oh_sets)", code)
            ),
            "trapezoidal_tau_integration": bool(
                re.search(r"(np\.trapezoid|np\.trapz|trapezoid\s*\(|trapz\s*\()", code)
            ),
            "npz_persistence": bool(
                re.search(r"(simulation_results\.npz|survival_fraction_results\.npz|np\.savez)", code)
            ),
            "consolidated_2x2_presentation": bool(
                re.search(r"(subplots\s*\(\s*2\s*,\s*2|full_analysis_presentation\.png|2x2)", code)
            ),
            "summary_report_txt": bool(
                re.search(r"(summary_report\.txt|report)", code)
            )
        }

    def evaluate(self, code: str, raw_response: str, notebook_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluates a script across the 5 flagship benchmark categories (100 pts total).
        Supports grounding against running.ipynb execution context (Anatase TiO2 / water).
        Strict mode: finer-grained sub-checks, tighter tolerances, severity-tiered caps.
        """
        if not code:
            return {
                "score": 0.0,
                "grade": "F",
                "extracted_code": "",
                "reference_notebook": self.REFERENCE_NOTEBOOK,
                "notebook_context": notebook_context,
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

        # =====================================================================
        # 1. Syntax & Core Imports (15 pts) [Notebook Cell 00 / running.ipynb]
        # =====================================================================
        cat1_score = 0
        has_syntax_error = False
        try:
            ast.parse(code)
            cat1_score += 4
            passed.append("Valid Python AST syntax (code compiles without syntax errors).")
        except SyntaxError as e:
            has_syntax_error = True
            failed.append(f"CRITICAL: Syntax error: {e.msg} at line {e.lineno}")

        # Specific ase.io import or pre-loaded traj integration from running.ipynb
        has_ase_io = bool(re.search(r"\b(from\s+ase\.io\s+import|from\s+ase\s+import\s+io|ase\.io\.read|read\s*\(|traj\b)", code))
        has_ase_generic = bool(re.search(r"\b(import\s+ase|from\s+ase|ase\b)\b", code))
        if has_ase_io:
            cat1_score += 3
            passed.append("ASE I/O or pre-loaded trajectory `traj` integration detected (Notebook Cell 00 / running.ipynb).")
        elif has_ase_generic:
            cat1_score += 1
            passed.append("Generic ASE import detected (missing specific ase.io submodule import).")
            failed.append("Missing specific `from ase.io import read` for trajectory loading (Notebook Cell 00).")
        else:
            failed.append("CRITICAL: Missing ASE import or `traj` usage entirely — cannot load atomic structures (Notebook Cell 00).")

        # Specific ase.neighborlist import
        has_ase_neighborlist = bool(re.search(r"\b(from\s+ase\.neighborlist|from\s+ase\s+import.*neighborlist|neighbor_list)", code))
        if has_ase_neighborlist:
            cat1_score += 2
            passed.append("Imported ASE neighborlist for coordination analysis (Notebook Cell 00).")
        else:
            failed.append("Missing `ase.neighborlist` import for neighbor search operations.")

        # NumPy import (Notebook Cell 00)
        has_numpy = bool(re.search(r"\b(import\s+numpy|from\s+numpy)\b", code))
        if has_numpy:
            cat1_score += 3
            passed.append("Imported NumPy for numerical operations and array manipulation (Notebook Cell 00).")
        else:
            failed.append("CRITICAL: Missing NumPy import (Notebook Cell 00).")

        # Matplotlib visualization imports (Notebook Cell 00/03/07)
        has_mpl = bool(re.search(r"\b(matplotlib|plt|pylab)\b", code))
        if has_mpl:
            cat1_score += 3
            passed.append("Imported Matplotlib for visualization and publication plots (Notebook Cell 00/03/07).")
        else:
            failed.append("Missing Matplotlib plotting imports.")

        # =====================================================================
        # 2. Interfacial Site Identification (20 pts) [Notebook Cell 02 & Cell 05]
        # =====================================================================
        cat2_score = 0

        # Surface metal identification with z-coordinate logic
        has_msur = bool(re.search(r"(m_sur|surface_metal|topmost|max\s*\([^)]*z|z\s*>\s*|z_coord|m_z|ti_sur|surface_ti|ti\b|titanium)", code, re.IGNORECASE))
        if has_msur:
            cat2_score += 3
            passed.append("Surface metal identification logic detected (identifies topmost metal layer, Notebook Cell 02).")
        else:
            failed.append("Missing surface metal layer identification logic (Notebook Cell 02 line 56).")

        # Z-coordinate sorting/selection (argsort, sort, argmax)
        has_z_sort = bool(re.search(r"(argsort|np\.sort|argmax|sorted.*z|z\[.*argsort|positions\[:.*2\])", code))
        if has_z_sort:
            cat2_score += 2
            passed.append("Z-coordinate sorting/selection for layer identification detected.")
        else:
            failed.append("Missing z-coordinate sorting (argsort/argmax) for systematic layer identification.")

        # Bulk/water oxygen classification via stoichiometry
        has_obulk = bool(re.search(r"(o_bul|bulk_oxygen|o_w|water_oxygen|stoichiomet|s\s*\*\s*len|s\s*\*\s*N_metal|o_bulk)", code, re.IGNORECASE))
        if has_obulk:
            cat2_score += 3
            passed.append("Bulk vs. water oxygen classification logic detected via oxide stoichiometry (Notebook Cell 02).")
        else:
            failed.append("Missing bulk vs water oxygen separation logic (Notebook Cell 02 lines 58-75).")

        # Stoichiometry ratio constant (s=2 for rutile MO2 / TiO2)
        has_stoich_ratio = bool(re.search(r"(s\s*=\s*2|stoich.*=\s*2|ratio\s*=\s*2|n_o.*2\s*\*|2\s*\*\s*n_metal|tio2)", code, re.IGNORECASE))
        if has_stoich_ratio:
            cat2_score += 2
            passed.append("Stoichiometry ratio s=2 for rutile MO₂/TiO₂ structure correctly specified.")
        else:
            failed.append("Missing explicit stoichiometry ratio (s=2 for rutile IrO₂/TiO₂).")

        # CUS classification
        has_cus = bool(re.search(r"\b(cus|m_cus|ti_cus|coordinatively\s+unsaturated)\b", code, re.IGNORECASE))
        if has_cus:
            cat2_score += 3
            passed.append("Coordinatively Unsaturated Site (CUS) classification detected (Notebook Cell 02).")
        else:
            failed.append("Missing CUS site classification logic (Notebook Cell 02 line 78).")

        # Explicit 2.5Å CUS cutoff distance
        has_cus_cutoff = bool(re.search(r"(2\.5|2\.50|cutoff.*2\.5|dist.*<=?\s*2\.5)", code))
        if has_cus_cutoff:
            cat2_score += 2
            passed.append("Explicit 2.5 Å distance cutoff for CUS oxygen coordination detected.")
        else:
            failed.append("Missing explicit 2.5 Å cutoff distance for CUS classification.")

        # Adsorbed/bridge oxygen tagging
        has_oadb_obr = bool(re.search(r"(o_adb|adsorbed|o_br|bridge|o_sur|o_inplane)", code, re.IGNORECASE))
        if has_oadb_obr:
            cat2_score += 3
            passed.append("Adsorbed oxygen (O_adb) and bridge oxygen (O_br) classification detected (Notebook Cell 02).")
        else:
            failed.append("Missing adsorbed/bridge oxygen site tagging (Notebook Cell 02 lines 79-94).")

        # Additional: explicit metal-oxygen distance check
        has_mo_dist = bool(re.search(r"(get_distance|get_all_distances|distance_matrix|cdist)", code))
        if has_mo_dist:
            cat2_score += 2
            passed.append("Pairwise distance computation for site classification detected.")
        else:
            failed.append("Missing explicit pairwise distance computation for site classification.")

        cat2_score = min(cat2_score, 20)

        # =====================================================================
        # 3. Density Profiles & Radial Distribution Functions (20 pts)
        # =====================================================================
        cat3_score = 0

        # Mass/number density profile
        has_density = bool(re.search(r"(density|mass_density|number_density|bins_z|histogram)", code, re.IGNORECASE))
        if has_density:
            cat3_score += 4
            passed.append("Mass density profile calculations across z-bins detected (Notebook Cell 02).")
        else:
            failed.append("Missing mass/number density profile calculations along z-axis.")

        # Bin-width specification (~0.1 Å)
        has_bin_width = bool(re.search(r"(dr\s*=\s*0\.\d|dz\s*=\s*0\.\d|bin_width|delta_z\s*=|nbins|linspace)", code, re.IGNORECASE))
        if has_bin_width:
            cat3_score += 2
            passed.append("Explicit bin-width/bin-count specification for density histogramming detected.")
        else:
            failed.append("Missing explicit bin-width specification (~0.1 Å) for density profiles.")

        # Unit conversion constants (18.015 g/mol, Avogadro's number)
        has_unit_conv = bool(re.search(r"(18\.015|6\.022e23|6\.023e23|avogadro|1e-24|1e24)", code, re.IGNORECASE))
        if has_unit_conv:
            cat3_score += 2
            passed.append("Physical unit conversion constants (18.015 g/mol, Avogadro) detected for density in g/cm³.")
        else:
            failed.append("Missing unit conversion constants (18.015, 6.022e23) for g/cm³ density.")

        # General RDF presence
        has_rdf = bool(re.search(r"(rdf|pair_distribution|g_r|rpdf)", code, re.IGNORECASE))
        if has_rdf:
            cat3_score += 3
            passed.append("Pair Radial Distribution Function (RDF) computation framework implemented (Notebook Cell 02).")
        else:
            failed.append("Missing Pair RDF (g(r)) calculations.")

        # Specific RDF pairs: M_cus-O_w
        has_rdf_mcus_ow = bool(re.search(r"(m_cus.*o_w|cus.*water|g_r_mow|metal.*water.*rdf|rdf.*cus|ti_cus.*o_w|ti.*water.*rdf|ti.*o_w)", code, re.IGNORECASE))
        if has_rdf_mcus_ow:
            cat3_score += 3
            passed.append("RDF pair: M_cus—O_w (metal CUS to water oxygen) detected.")
        else:
            failed.append("Missing specific RDF pair: M_cus—O_w (metal CUS to water oxygen).")

        # Specific RDF pairs: O_br-H_w
        has_rdf_obr_hw = bool(re.search(r"(o_br.*h_w|bridge.*hydrogen|g_r_osh|obr.*hw|rdf.*bridge.*h)", code, re.IGNORECASE))
        if has_rdf_obr_hw:
            cat3_score += 3
            passed.append("RDF pair: O_br—H_w (bridge oxygen to water hydrogen) detected.")
        else:
            failed.append("Missing specific RDF pair: O_br—H_w (bridge oxygen to water hydrogen).")

        # Specific RDF pairs: O_br-O_w
        has_rdf_obr_ow = bool(re.search(r"(o_br.*o_w|bridge.*water.*o|g_r_osow|obr.*ow|rdf.*bridge.*o)", code, re.IGNORECASE))
        if has_rdf_obr_ow:
            cat3_score += 3
            passed.append("RDF pair: O_br—O_w (bridge oxygen to water oxygen) detected.")
        else:
            failed.append("Missing specific RDF pair: O_br—O_w (bridge oxygen to water oxygen).")

        cat3_score = min(cat3_score, 20)

        # =====================================================================
        # 4. Surface Kinetics & Survival Probability (25 pts) [Cell 05 & 06]
        # =====================================================================
        cat4_score = 0

        # Hydroxyl tracking with 1.2 Å cutoff
        has_hydroxyl = bool(re.search(r"(hydroxyl|dissociat|frac_oh|surv_oh|neighbor_list|cutoff.*1\.[1-5]|indx_oh)", code, re.IGNORECASE))
        if has_hydroxyl:
            cat4_score += 6
            passed.append("Surface hydroxylation / water dissociation tracking implemented (Notebook Cell 05).")
        else:
            failed.append("Missing surface hydroxyl fraction / dissociation calculation (Notebook Cell 05).")

        # Survival probability P(t)
        has_survival = bool(re.search(r"(survival|p_t|residence|time_lag|lag|intersection|tau)", code, re.IGNORECASE))
        if has_survival:
            cat4_score += 6
            passed.append("Interfacial water survival probability P(t) tracking implemented (Notebook Cell 05/06).")
        else:
            failed.append("Missing water survival probability P(t) calculation (Notebook Cell 05).")

        # Log-spaced time lags
        has_log_lags = bool(re.search(r"(logspace|geomspace|log_spaced|np\.log|log.*lag|geometric)", code, re.IGNORECASE))
        if has_log_lags:
            cat4_score += 4
            passed.append("Log-spaced time lags for survival probability computation detected.")
        else:
            failed.append("Missing log-spaced time lags (logspace/geomspace) for P(t) sampling.")

        # Continuous set intersection algorithm
        has_set_intersection = bool(re.search(r"(set\.intersection|\.intersection\s*\(|oh_sets|set\s*&\s*set|&=)", code))
        if has_set_intersection:
            cat4_score += 4
            passed.append("Continuous set intersection algorithm for survival tracking detected.")
        else:
            failed.append("Missing continuous set intersection algorithm for survival probability.")

        # Trapezoidal integration for residence time tau
        has_trapz = bool(re.search(r"(np\.trapezoid|np\.trapz|trapezoid\s*\(|trapz\s*\(|scipy\.integrate)", code))
        if has_trapz:
            cat4_score += 5
            passed.append("Trapezoidal integration for residence time τ computation detected.")
        else:
            failed.append("Missing trapezoidal integration (np.trapz/trapezoid) for τ calculation.")

        cat4_score = min(cat4_score, 25)

        # =====================================================================
        # 5. DP Model Averaging & Multi-Panel Artifacts (20 pts) [Cell 02/05/07]
        # =====================================================================
        cat5_score = 0

        # Multi-model or trajectory frame loop
        has_multi_dp = bool(re.search(r"(dp_model|models|mean|std|average|cross_model|model_idx|modelz|model_num|traj|frame)", code, re.IGNORECASE))
        if has_multi_dp:
            cat5_score += 3
            passed.append("Multi-model trajectory loop / cross-model averaging (mean ± std) implemented.")
        else:
            failed.append("Missing multi-model DP loop or cross-model averaging logic.")

        # Individual plots
        has_indiv_plots = bool(re.search(r"(savefig|plot.*density|plot.*rdf|plot.*survival|fig1|fig2|plt\.show)", code, re.IGNORECASE))
        if has_indiv_plots:
            cat5_score += 3
            passed.append("Individual plot generation and file saving logic detected.")
        else:
            failed.append("Missing individual figure saving routines.")

        # Specific figure filenames (density, rdf, survival)
        has_specific_figs = bool(re.search(r"(density.*\.png|rdf.*\.png|survival.*\.png|fraction.*\.png)", code, re.IGNORECASE))
        if has_specific_figs:
            cat5_score += 2
            passed.append("Domain-specific figure filenames (density.png, rdf.png, survival.png) detected.")
        else:
            failed.append("Missing domain-specific figure filenames for individual plots.")

        # 2x2 multi-panel summary figure
        has_2x2 = bool(re.search(r"(subplots\s*\(\s*2\s*,\s*2|2x2|2,\s*2|full_analysis_presentation)", code, re.IGNORECASE))
        if has_2x2:
            cat5_score += 3
            passed.append("Combined 2×2 multi-panel summary figure layout implemented (Notebook Cell 07).")
        else:
            failed.append("Missing combined 2×2 summary figure creation (e.g. plt.subplots(2, 2)).")

        # .npz data persistence
        has_npz = bool(re.search(r"(np\.savez|\.npz|savez_compressed)", code))
        if has_npz:
            cat5_score += 3
            passed.append("NumPy .npz data persistence (np.savez) for reproducibility detected.")
        else:
            failed.append("Missing .npz data persistence (np.savez) for result reproducibility.")

        # Text summary report
        has_report = bool(re.search(r"(summary_report\.txt|summary_report)", code, re.IGNORECASE))
        if has_report:
            cat5_score += 3
            passed.append("Text summary report generation (summary_report.txt) implemented.")
        else:
            failed.append("Missing summary_report.txt output generation.")

        # Report content validation (residence time + fractions mentioned)
        has_report_content = bool(re.search(r"(residence.*time|tau.*write|frac.*write|dissociat.*write|report.*tau|report.*frac)", code, re.IGNORECASE))
        if has_report_content:
            cat5_score += 3
            passed.append("Summary report contains residence times and dissociation fractions.")
        else:
            failed.append("Summary report missing required content (residence times + dissociation fractions).")

        cat5_score = min(cat5_score, 20)

        # =====================================================================
        # Total Score & Grade (Strict Severity-Tiered Caps)
        # =====================================================================
        raw_total = cat1_score + cat2_score + cat3_score + cat4_score + cat5_score
        raw_total = min(max(round(raw_total, 1), 0.0), 100.0)
        total_score = raw_total

        # Count critical failures
        critical_failures = sum(1 for f in failed if f.startswith("CRITICAL:"))

        if has_syntax_error:
            # CRITICAL: Syntax error caps at 35 (stricter than previous 45)
            total_score = min(total_score, 35.0)
            grade = "F"
            summary = "CRITICAL FAILURE: Script contains a fatal syntax error and cannot be compiled."
        elif critical_failures >= 2:
            # Multiple critical failures (missing core imports)
            total_score = min(total_score, 50.0)
            if total_score >= 50:
                grade = "D"
            else:
                grade = "F"
            summary = f"Script has {critical_failures} critical failures (missing core imports/libraries)."
        elif total_score >= 90:
            grade = "A"
        elif total_score >= 80:
            grade = "B"
        elif total_score >= 70:
            grade = "C"
        elif total_score >= 60:
            grade = "D"
        else:
            grade = "F"

        if not has_syntax_error and critical_failures < 2:
            summary = (
                f"Zero-shot script achieved an accuracy score of {total_score}/100 (Grade {grade}) "
                f"against the reference notebook standard. Passed {len(passed)}/{len(passed) + len(failed)} scientific checks. "
            )
            if total_score >= 90:
                summary += "Outstanding compliance: The generated code faithfully replicates all stages of the reference analysis pipeline."
            elif total_score >= 80:
                summary += "The model demonstrated strong scientific comprehension of interfacial water workflows from the notebook."
            elif total_score >= 70:
                summary += "Solid implementation with minor gaps in specific sub-checks or algorithmic details."
            elif total_score >= 60:
                summary += "The model grasped the general pipeline structure but missed key mathematical or site classification details."
            else:
                summary += "The model struggled with zero-shot generation of the complete multi-step scientific analysis pipeline."

        notebook_features = self.inspect_notebook_features(code)
        extracted_cand = self.extract_metrics(raw_response)
        metric_comparison = self.compare_metrics(extracted_cand) if extracted_cand else None

        category_scores = {
            "syntax_and_imports": {"score": cat1_score, "max": 15, "pct": round(cat1_score / 15 * 100)},
            "site_identification": {"score": cat2_score, "max": 20, "pct": round(cat2_score / 20 * 100)},
            "density_and_rdfs": {"score": cat3_score, "max": 20, "pct": round(cat3_score / 20 * 100)},
            "kinetics_and_survival": {"score": cat4_score, "max": 25, "pct": round(cat4_score / 25 * 100)},
            "model_averaging_and_outputs": {"score": cat5_score, "max": 20, "pct": round(cat5_score / 20 * 100)}
        }

        # Scale category scores if overall score was capped/penalized so breakdown matches total
        if raw_total > 0 and total_score < raw_total:
            category_scores = self.scale_category_scores(category_scores, total_score)

        return {
            "score": total_score,
            "grade": grade,
            "extracted_code": code,
            "reference_notebook": self.REFERENCE_NOTEBOOK,
            "category_scores": category_scores,
            "passed_checks": passed,
            "failed_checks": failed,
            "notebook_features": notebook_features,
            "metric_comparison": metric_comparison,
            "summary": summary
        }

    @staticmethod
    def scale_category_scores(
        category_scores: Dict[str, Dict[str, Any]],
        target_total: float
    ) -> Dict[str, Dict[str, Any]]:
        """
        Scales individual category scores proportionally so their sum strictly equals target_total.
        Prevents misleading displays where category bars show 100% or contradict a capped total score.
        """
        raw_sum = sum(c["score"] for c in category_scores.values())
        if raw_sum <= 0 or abs(raw_sum - target_total) < 0.001:
            if target_total == 0:
                return {k: {"score": 0, "max": v["max"], "pct": 0} for k, v in category_scores.items()}
            return category_scores

        scale = target_total / raw_sum
        adjusted: Dict[str, Dict[str, Any]] = {}
        alloc_sum = 0
        keys = list(category_scores.keys())

        for k in keys:
            orig = category_scores[k]
            scaled = round(orig["score"] * scale)
            scaled = min(scaled, orig["max"])
            adjusted[k] = {
                "score": int(scaled),
                "max": orig["max"],
                "pct": min(100, round((scaled / orig["max"]) * 100)) if orig["max"] > 0 else 0
            }
            alloc_sum += scaled

        remainder = int(round(target_total - alloc_sum))
        if remainder != 0 and keys:
            largest_key = max(keys, key=lambda k: adjusted[k]["score"])
            adjusted[largest_key]["score"] += remainder
            max_v = adjusted[largest_key]["max"]
            adjusted[largest_key]["pct"] = min(100, round((adjusted[largest_key]["score"] / max_v) * 100))

        return adjusted
