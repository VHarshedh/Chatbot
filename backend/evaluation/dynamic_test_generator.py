import ast
import re
import os
import sys
import subprocess
import tempfile
from typing import Dict, Any, List, Optional, Set
from .ast_analyzer import AstAnalyzer
from .domain_rules import DomainClassifier, ScientificDomain, DomainRule

class DynamicTestGenerator:
    """
    Generates and executes dynamic verification suites for computational chemistry
    and scientific simulation scripts outside the pre-indexed RAG benchmark.
    
    Operates locally and deterministically using:
    1. AST contract verification and undefined symbol / missing import detection.
    2. Domain-specific physical invariant rules.
    3. Scientific rigor assessment (units, naming, docstrings, error handling).
    4. Isolated runtime sandbox execution testing to catch unhandled exceptions
       (NameError, ValueError, TypeError, ZeroDivisionError, etc.).
    5. Import completeness verification (unused imports penalty).
    """

    # Tightened from 5s to 3s
    SANDBOX_TIMEOUT: int = 3

    def __init__(self, prompt: Optional[str] = None):
        self.prompt = prompt or ""

    @staticmethod
    def run_sandbox_execution(code: str, timeout: int = 3) -> Dict[str, Any]:
        """
        Executes the candidate Python script in an isolated temporary directory
        with an enforced timeout. Captures exit code, stdout, stderr, and any
        unhandled runtime exceptions.
        """
        if not code or not code.strip():
            return {"executed": False, "passed": False, "error": "Empty code", "stderr": "", "stdout": "", "artifacts": []}

        with tempfile.TemporaryDirectory(prefix="eval_sandbox_") as tmp_dir:
            script_file = os.path.join(tmp_dir, "candidate_script.py")
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(code)

            try:
                proc = subprocess.run(
                    [sys.executable, script_file],
                    cwd=tmp_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                artifacts = [
                    f for f in os.listdir(tmp_dir)
                    if f != "candidate_script.py" and not f.startswith("__")
                ]

                if proc.returncode != 0:
                    err_lines = [l.strip() for l in proc.stderr.splitlines() if l.strip()]
                    exception_line = err_lines[-1] if err_lines else f"Exited with code {proc.returncode}"
                    return {
                        "executed": True,
                        "passed": False,
                        "returncode": proc.returncode,
                        "error": exception_line,
                        "stderr": proc.stderr,
                        "stdout": proc.stdout,
                        "artifacts": artifacts
                    }

                # Also inspect stdout for printed exception tracebacks (e.g. within try/except handlers in the code)
                if "An error occurred during execution:" in proc.stdout or "Traceback (most recent call last):" in proc.stdout:
                    err_lines = [l.strip() for l in proc.stdout.splitlines() if "Error" in l or "error" in l]
                    err_summary = err_lines[-1] if err_lines else "Caught runtime exception in script execution"
                    return {
                        "executed": True,
                        "passed": False,
                        "returncode": 1,
                        "error": err_summary,
                        "stderr": proc.stderr,
                        "stdout": proc.stdout,
                        "artifacts": artifacts
                    }

                return {
                    "executed": True,
                    "passed": True,
                    "returncode": 0,
                    "error": None,
                    "stderr": proc.stderr,
                    "stdout": proc.stdout,
                    "artifacts": artifacts
                }

            except subprocess.TimeoutExpired:
                return {
                    "executed": True,
                    "passed": False,
                    "returncode": -1,
                    "error": f"Execution timed out (exceeded {timeout}s limit)",
                    "stderr": "",
                    "stdout": "",
                    "artifacts": []
                }
            except Exception as e:
                return {
                    "executed": False,
                    "passed": False,
                    "returncode": -1,
                    "error": str(e),
                    "stderr": str(e),
                    "stdout": "",
                    "artifacts": []
                }

    @staticmethod
    def check_scientific_rigor(code: str, ast_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assesses scientific rigor: unit annotations, naming conventions,
        docstrings, and error handling.
        """
        rigor_score = 0
        rigor_checks_passed: List[str] = []
        rigor_checks_failed: List[str] = []

        # 1. Physical unit annotations in comments (5 pts)
        unit_patterns = [
            r"#.*\b(Å|angstrom|nm|pm|bohr)\b",
            r"#.*\b(eV|kcal|kJ|hartree|Ry)\b",
            r"#.*\b(g/cm|kg/m|mol|atoms)\b",
            r"#.*\b(ps|fs|ns|μs)\b",
            r"#.*\b(K|kelvin|GPa|bar|atm)\b",
            r"#.*\bunits?\b",
        ]
        unit_matches = sum(1 for p in unit_patterns if re.search(p, code, re.IGNORECASE))
        if unit_matches >= 2:
            rigor_score += 5
            rigor_checks_passed.append("Physical unit annotations found in code comments (≥2 unit types).")
        elif unit_matches == 1:
            rigor_score += 2
            rigor_checks_passed.append("Partial physical unit annotations found in comments (1 unit type).")
            rigor_checks_failed.append("Insufficient unit annotations — scientific code should document physical units.")
        else:
            rigor_checks_failed.append("Missing physical unit annotations in comments — scientific code must document units.")

        # 2. Meaningful variable names (4 pts) — penalize excessive single-letter variables
        single_letter_vars: Set[str] = set()
        if ast_info.get("syntax_valid") and "tree" not in ast_info:
            # Re-parse to check
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        if len(node.id) == 1 and node.id not in {"_", "i", "j", "k", "x", "y", "z", "r", "f"}:
                            single_letter_vars.add(node.id)
            except Exception:
                pass
        
        if len(single_letter_vars) <= 1:
            rigor_score += 4
            rigor_checks_passed.append("Variable naming conventions follow best practices (meaningful names).")
        elif len(single_letter_vars) <= 3:
            rigor_score += 2
            rigor_checks_failed.append(f"Some cryptic single-letter variable names detected: {', '.join(sorted(single_letter_vars)[:3])}")
        else:
            rigor_checks_failed.append(f"Excessive single-letter variables ({len(single_letter_vars)}): {', '.join(sorted(single_letter_vars)[:5])}. Use descriptive names.")

        # 3. Docstrings on functions (3 pts)
        functions = ast_info.get("functions", [])
        if functions:
            funcs_with_docs = sum(1 for fn in functions if fn.get("docstring"))
            doc_ratio = funcs_with_docs / len(functions) if len(functions) > 0 else 0
            if doc_ratio >= 0.5:
                rigor_score += 3
                rigor_checks_passed.append(f"Functions are documented ({funcs_with_docs}/{len(functions)} have docstrings).")
            elif doc_ratio > 0:
                rigor_score += 1
                rigor_checks_failed.append(f"Insufficient documentation — only {funcs_with_docs}/{len(functions)} functions have docstrings.")
            else:
                rigor_checks_failed.append("No function docstrings found. Scientific code should be self-documenting.")
        else:
            # No functions defined — check for top-level comments at least
            comment_lines = sum(1 for line in code.splitlines() if line.strip().startswith("#"))
            if comment_lines >= 5:
                rigor_score += 2
                rigor_checks_passed.append("Script-level comments provide documentation (no functions defined).")
            else:
                rigor_checks_failed.append("No functions or meaningful comments — script lacks documentation.")

        # 4. Error handling (3 pts)
        has_try_except = bool(re.search(r"\btry\s*:", code))
        has_assertions = bool(re.search(r"\bassert\b", code))
        has_validation = bool(re.search(r"(if\s+.*(?:is\s+None|len\s*\(|not\s+|==\s*0|<\s*0))", code))
        
        error_handling_score = 0
        if has_try_except:
            error_handling_score += 2
        if has_assertions or has_validation:
            error_handling_score += 1
        
        if error_handling_score >= 2:
            rigor_score += 3
            rigor_checks_passed.append("Proper error handling with try/except and input validation detected.")
        elif error_handling_score == 1:
            rigor_score += 1
            rigor_checks_failed.append("Minimal error handling — add try/except for robustness.")
        else:
            rigor_checks_failed.append("No error handling detected — scientific code should handle edge cases gracefully.")

        return {
            "score": min(rigor_score, 15),
            "max": 15,
            "passed": rigor_checks_passed,
            "failed": rigor_checks_failed
        }

    @staticmethod
    def check_import_completeness(code: str, ast_info: Dict[str, Any]) -> Dict[str, Any]:
        """Detects imported-but-unused modules as a code quality indicator."""
        imports = ast_info.get("imports", {})
        all_imported = imports.get("modules", set()) | imports.get("aliases", set())
        
        # Simple heuristic: check if imported names appear in the code beyond import lines
        unused_imports: List[str] = []
        code_lines = code.splitlines()
        non_import_code = "\n".join(
            line for line in code_lines 
            if not line.strip().startswith(("import ", "from ")) and not line.strip().startswith("#")
        )
        
        for imp in all_imported:
            if imp and not re.search(r"\b" + re.escape(imp) + r"\b", non_import_code):
                unused_imports.append(imp)
        
        return {
            "unused_imports": unused_imports,
            "all_used": len(unused_imports) == 0
        }

    def generate_and_evaluate(self, code: str, ast_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Synthesizes modular test cases, validates symbol scopes, executes sandbox tests,
        assesses scientific rigor, and generates a rigorous evaluation report.
        
        Categories (100 pts):
          1. Syntax & Safety: 15 pts
          2. Domain Implementation: 30 pts  
          3. Runtime Execution & Contracts: 25 pts
          4. Scientific Rigor: 15 pts
          5. Artifacts & Presentation: 15 pts
        """
        if not code or not code.strip():
            return {
                "score": 0.0,
                "grade": "F",
                "domain": ScientificDomain.GENERAL_SIMULATION.value,
                "category_scores": {
                    "syntax_and_safety": {"score": 0, "max": 15, "pct": 0},
                    "domain_implementation": {"score": 0, "max": 30, "pct": 0},
                    "runtime_execution_and_contracts": {"score": 0, "max": 25, "pct": 0},
                    "scientific_rigor": {"score": 0, "max": 15, "pct": 0},
                    "artifacts_and_presentation": {"score": 0, "max": 15, "pct": 0}
                },
                "passed_checks": [],
                "failed_checks": ["No code found to evaluate."],
                "generated_tests": [],
                "summary": "Evaluation failed: Empty script content provided."
            }

        # Clean markdown code fences if provided directly
        if "```" in code:
            m = re.search(r"```(?:python|py)?\s*([\s\S]*?)```", code, re.IGNORECASE)
            if m:
                code = m.group(1).strip()
            else:
                code = re.sub(r"^```[^\n]*\n", "", code)
                code = re.sub(r"```$", "", code).strip()

        if ast_info is None:
            analyzer = AstAnalyzer(code)
            ast_info = analyzer.analyze()

        domain = DomainClassifier.classify(self.prompt, code)
        rules = DomainClassifier.get_rules_for_domain(domain)

        passed_checks: List[str] = []
        failed_checks: List[str] = []
        generated_tests: List[Dict[str, Any]] = []

        # ─────────────────────────────────────────────────────────────────────
        # 1. Syntax & Safety Category (15 pts)
        # ─────────────────────────────────────────────────────────────────────
        cat_syntax_score = 0
        syntax_valid = ast_info.get("syntax_valid", False)
        if syntax_valid:
            cat_syntax_score += 8
            passed_checks.append("Valid Python AST syntax (code compiles without syntax errors).")
            generated_tests.append({"name": "test_syntax_compilation", "passed": True, "details": "AST parsed successfully."})
        else:
            err_msg = ast_info.get("syntax_error", "Unknown syntax error")
            failed_checks.append(f"CRITICAL: Syntax error in script: {err_msg}")
            generated_tests.append({"name": "test_syntax_compilation", "passed": False, "details": err_msg})

        safety_info = ast_info.get("safety", {"safe": True, "violations": []})
        if safety_info.get("safe", True):
            cat_syntax_score += 7
            passed_checks.append("Code passed security isolation audit (no unauthorized subshells or destructive calls).")
            generated_tests.append({"name": "test_safety_audit", "passed": True, "details": "No unsafe system calls detected."})
        else:
            violations = ", ".join(safety_info.get("violations", []))
            failed_checks.append(f"Security audit failed: {violations}")
            generated_tests.append({"name": "test_safety_audit", "passed": False, "details": violations})

        # ─────────────────────────────────────────────────────────────────────
        # 2. Domain Implementation Category (30 pts)
        # ─────────────────────────────────────────────────────────────────────
        cat_domain_score = 0
        total_rule_weight = sum(r.weight for r in rules) or 100

        for rule in rules:
            is_passed = rule.evaluate(code, ast_info)
            rule_pts = round((rule.weight / total_rule_weight) * 30)
            if is_passed:
                cat_domain_score += rule_pts
                passed_checks.append(f"[{domain.value.upper()}] {rule.description}")
                generated_tests.append({"name": f"test_{rule.rule_id}", "passed": True, "details": rule.description})
            else:
                failed_checks.append(f"[{domain.value.upper()}] Missing requirement: {rule.description}")
                generated_tests.append({"name": f"test_{rule.rule_id}", "passed": False, "details": rule.description})

        cat_domain_score = min(cat_domain_score, 30)

        # ─────────────────────────────────────────────────────────────────────
        # 3. Scope Resolution & Runtime Execution (25 pts)
        # ─────────────────────────────────────────────────────────────────────
        cat_exec_score = 0

        # Scope Check: Undefined variables or missing imports (8 pts)
        undefined_symbols = ast_info.get("undefined_symbols", [])
        if not undefined_symbols:
            cat_exec_score += 8
            passed_checks.append("Scope resolution verified (all referenced symbols and modules are properly imported or defined).")
            generated_tests.append({"name": "test_scope_resolution", "passed": True, "details": "No undefined symbols."})
        else:
            missing_str = ", ".join(undefined_symbols[:5])
            failed_checks.append(f"Scope error: Undefined variables or missing imports detected: {missing_str}")
            generated_tests.append({"name": "test_scope_resolution", "passed": False, "details": f"Undefined: {missing_str}"})

        # Import completeness check (2 pts)
        import_check = self.check_import_completeness(code, ast_info)
        if import_check["all_used"]:
            cat_exec_score += 2
            passed_checks.append("All imported modules are actively used in the code.")
            generated_tests.append({"name": "test_import_completeness", "passed": True, "details": "No unused imports."})
        else:
            unused = ", ".join(import_check["unused_imports"][:3])
            failed_checks.append(f"Unused imports detected: {unused} — clean imports improve maintainability.")
            generated_tests.append({"name": "test_import_completeness", "passed": False, "details": f"Unused: {unused}"})

        # Sandboxed Runtime Execution (15 pts)
        exec_result = self.run_sandbox_execution(code, timeout=self.SANDBOX_TIMEOUT)
        if exec_result.get("passed", False):
            cat_exec_score += 15
            passed_checks.append("Sandboxed runtime execution passed (script ran without unhandled exceptions).")
            generated_tests.append({"name": "test_sandbox_execution", "passed": True, "details": "Exit code 0."})
        else:
            err_detail = exec_result.get("error", "Runtime crash")
            failed_checks.append(f"Runtime execution failure: {err_detail}")
            generated_tests.append({"name": "test_sandbox_execution", "passed": False, "details": err_detail})

        # ─────────────────────────────────────────────────────────────────────
        # 4. Scientific Rigor (15 pts)
        # ─────────────────────────────────────────────────────────────────────
        rigor_result = self.check_scientific_rigor(code, ast_info)
        cat_rigor_score = rigor_result["score"]
        passed_checks.extend(rigor_result["passed"])
        failed_checks.extend(rigor_result["failed"])

        # ─────────────────────────────────────────────────────────────────────
        # 5. Artifacts, Visualization & Output (15 pts)
        # ─────────────────────────────────────────────────────────────────────
        cat_artifacts_score = 0
        patterns = ast_info.get("patterns", {})
        has_plot_artifact = len(exec_result.get("artifacts", [])) > 0 or patterns.get("has_plotting", False)
        if has_plot_artifact:
            cat_artifacts_score += 8
            passed_checks.append("Visualization routine implemented (matplotlib plots / saved figure artifacts).")
        elif bool(re.search(r"(plt\.|plot|scatter|figure)", code, re.IGNORECASE)):
            cat_artifacts_score += 4
            passed_checks.append("Plotting calls detected in candidate script.")
        else:
            failed_checks.append("Missing graphical visualization or figure generation routines.")

        if patterns.get("has_saving", False) or patterns.get("has_file_io", False) or "print(" in code:
            cat_artifacts_score += 7
            passed_checks.append("Result persistence or summary reporting logic verified.")
        else:
            failed_checks.append("Missing output file persistence or terminal reporting.")

        # Code complexity penalty: excessive nested loops
        loop_count = patterns.get("loop_count", 0)
        if loop_count >= 5:
            complexity_penalty = min(3, loop_count - 4)
            cat_artifacts_score = max(0, cat_artifacts_score - complexity_penalty)
            failed_checks.append(f"Performance warning: Excessive non-vectorized loops ({loop_count} detected, -{complexity_penalty}pt penalty).")
            generated_tests.append({"name": "test_vectorization_efficiency", "passed": False, "details": f"{loop_count} loops, -{complexity_penalty}pt"})
        elif loop_count >= 4:
            failed_checks.append(f"Performance advisory: Relies on nested loops ({loop_count} loops detected). Consider vectorization.")
            generated_tests.append({"name": "test_vectorization_efficiency", "passed": False, "details": f"{loop_count} nested loops detected"})

        # ─────────────────────────────────────────────────────────────────────
        # Total Score & Grade Calculation with Strict Penalties
        # ─────────────────────────────────────────────────────────────────────
        raw_total = cat_syntax_score + cat_domain_score + cat_exec_score + cat_rigor_score + cat_artifacts_score
        raw_total = min(max(round(raw_total, 1), 0.0), 100.0)
        total_score = raw_total

        # Enforce strict grade caps:
        has_runtime_crash = not exec_result.get("passed", False)
        has_scope_error = len(undefined_symbols) > 0

        if not syntax_valid:
            total_score = min(total_score, 35.0)
            grade = "F"
            summary = f"CRITICAL FAILURE: Script contains a fatal syntax error ({ast_info.get('syntax_error', 'syntax error')}) and cannot be compiled."
        elif has_runtime_crash:
            # Stricter: cap at 45 instead of 55
            total_score = min(total_score, 45.0)
            grade = "D" if total_score >= 40 else "F"
            summary = f"Evaluation degraded: Script compiled but failed runtime execution ({exec_result.get('error')})."
        elif has_scope_error:
            total_score = min(total_score, 60.0)
            grade = "D"
            missing_str = ", ".join(undefined_symbols[:3])
            summary = f"Evaluation degraded: Script references undefined variables or missing imports ({missing_str})."
        else:
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
                f"Dynamic hybrid evaluation for domain [{domain.value}] achieved a score of {total_score}/100 (Grade {grade}). "
                f"Passed {len(passed_checks)}/{len(passed_checks) + len(failed_checks)} generated test assertions."
            )

        category_scores = {
            "syntax_and_safety": {"score": cat_syntax_score, "max": 15, "pct": round(cat_syntax_score / 15 * 100)},
            "domain_implementation": {"score": cat_domain_score, "max": 30, "pct": round(cat_domain_score / 30 * 100)},
            "runtime_execution_and_contracts": {"score": cat_exec_score, "max": 25, "pct": round(cat_exec_score / 25 * 100)},
            "scientific_rigor": {"score": cat_rigor_score, "max": 15, "pct": round(cat_rigor_score / 15 * 100)},
            "artifacts_and_presentation": {"score": cat_artifacts_score, "max": 15, "pct": round(cat_artifacts_score / 15 * 100)}
        }

        # Scale category scores if overall score was capped/penalized so breakdown matches total
        if raw_total > 0 and total_score < raw_total:
            category_scores = self.scale_category_scores(category_scores, total_score)

        return {
            "score": total_score,
            "grade": grade,
            "domain": domain.value,
            "category_scores": category_scores,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "generated_tests": generated_tests,
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
