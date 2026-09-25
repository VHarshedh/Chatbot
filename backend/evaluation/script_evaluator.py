import re
from typing import Dict, Any, List, Optional
from .benchmark_evaluator import BenchmarkEvaluator
from .dynamic_test_generator import DynamicTestGenerator
from .ast_analyzer import AstAnalyzer

class ScriptEvaluator:
    """
    Unified Hybrid Scientific Script Evaluation Engine.
    
    Provides dual-path evaluation:
    1. RAG Flagship Benchmark: 100-point ground truth standard and numerical metric tolerance
       comparison against Water_Density_RDF_Survival_Probability...ipynb.
    2. Out-of-RAG Dynamic Hybrid: Modular, purely local dynamic test generation for arbitrary
       computational materials science, molecular dynamics, DFT, and surface science workflows.
    """

    REFERENCE_NOTEBOOK: str = BenchmarkEvaluator.REFERENCE_NOTEBOOK
    REFERENCE_METRICS: Dict[str, Dict[str, Any]] = BenchmarkEvaluator.REFERENCE_METRICS

    FLAGSHIP_BENCHMARK_KEYWORDS: List[str] = [
        "interfacial", "rutile", "iro2", "anatase", "running.ipynb", "water_density", "survival_probability",
        "hydroxyl_fraction", "flagship", "100-point", "ground truth standard"
    ]

    def __init__(self):
        self._benchmark_evaluator = BenchmarkEvaluator()

    @staticmethod
    def extract_python_code(raw_text: str) -> str:
        """Extracts Python code blocks from markdown-formatted LLM responses."""
        if not raw_text:
            return ""

        # 1. Strip out thinking details blocks so HTML details never contaminate Python code
        cleaned_text = re.sub(r"<details[\s\S]*?<\/details>", "", raw_text, flags=re.IGNORECASE).strip()

        # 2. Match fenced code blocks with language tag and filter for python
        raw_blocks = re.findall(r"```([a-zA-Z0-9_-]*)\s*?\n?([\s\S]*?)(?:```|$)", cleaned_text)
        code_candidates = []
        for lang, body in raw_blocks:
            lang_clean = lang.strip().lower()
            if lang_clean in ("", "python", "py"):
                code_candidate = body.strip()
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
        """Parses numerical metrics (tau_oh, tau_h2o, frac_oh, frac_h2o) from text."""
        return BenchmarkEvaluator.extract_metrics(text)

    def compare_metrics(
        self,
        candidate_metrics: Dict[str, float],
        tolerance: float = 0.05
    ) -> Dict[str, Any]:
        """Compares candidate numerical metrics against ground-truth values."""
        return self._benchmark_evaluator.compare_metrics(candidate_metrics, tolerance=tolerance)

    def inspect_notebook_features(self, code: str) -> Dict[str, Any]:
        """Detects specific algorithmic features from the flagship notebook."""
        return self._benchmark_evaluator.inspect_notebook_features(code)

    def is_flagship_benchmark(self, prompt: Optional[str], code: str) -> bool:
        """
        Determines whether the target script corresponds to the pre-indexed
        flagship interfacial water analysis benchmark.
        """
        # If no prompt provided, default to flagship benchmark for backward compatibility
        if not prompt or not prompt.strip():
            return True

        lower_prompt = prompt.lower()
        if any(kw in lower_prompt for kw in self.FLAGSHIP_BENCHMARK_KEYWORDS):
            return True

        # Check for signature benchmark variables/structures in code
        if code and re.search(r"(m_sur|o_bulk|cus_sites|full_analysis_presentation|survival_fraction_results|anatase_wrapped|ti_cus)", code):
            return True

        return False

    def evaluate(
        self,
        raw_response: str,
        prompt: Optional[str] = None,
        notebook_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a zero-shot or agentic generated script.
        Routes to the RAG Flagship Benchmark Evaluator or the Dynamic Test Generator
        based on prompt context and code structure.
        """
        code = self.extract_python_code(raw_response)

        # Flagship RAG Benchmark Route
        if self.is_flagship_benchmark(prompt, code):
            result = self._benchmark_evaluator.evaluate(code, raw_response, notebook_context=notebook_context)
            result["evaluation_type"] = "rag_benchmark"
            return result

        # Out-of-RAG Dynamic Hybrid Route
        test_gen = DynamicTestGenerator(prompt=prompt)
        dyn_result = test_gen.generate_and_evaluate(code)

        return {
            "score": dyn_result["score"],
            "grade": dyn_result["grade"],
            "extracted_code": code,
            "evaluation_type": "dynamic_hybrid",
            "domain": dyn_result.get("domain", "general_simulation"),
            "reference_notebook": None,
            "notebook_context": notebook_context,
            "category_scores": dyn_result["category_scores"],
            "passed_checks": dyn_result["passed_checks"],
            "failed_checks": dyn_result["failed_checks"],
            "generated_tests": dyn_result.get("generated_tests", []),
            "notebook_features": {},
            "metric_comparison": None,
            "summary": dyn_result["summary"]
        }
