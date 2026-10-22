import os
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import requests
from dotenv import load_dotenv
from backend.config import settings
from backend.db.models import insert_evaluation

# Ensure .env is loaded
env_file = Path(settings.WORKSPACE_ROOT) / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    load_dotenv()


class GeminiEvaluator:
    """
    Strict Scientific Code Evaluator powered by Gemini 3.8 Flash.
    
    Evaluates generated scientific Python scripts with extreme prejudice,
    strict mathematical/physical rigor, and a 100-point rubric.
    Incorrect outputs or failing code are severely penalized around 20%.
    """

    PRIMARY_MODEL = getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash"
    FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest"]

    STRICT_SYSTEM_PROMPT = (
        "You are an ultra-strict, ruthless scientific code auditor and senior computational materials scientist.\n"
        "You evaluate generated scientific Python code with EXTREME PREJUDICE, ZERO LENIENCY, and ABSOLUTE RIGOR.\n"
        "You grade strictly on a 0 to 100 point scale.\n\n"
        "CRITICAL SCORING RULES:\n"
        "1. INCORRECT OUTPUT OR CODE FAILURE: If the code has syntax errors, runtime exceptions, missing variables, "
        "invalid logic, hallucinations (such as fabricated ASE, NumPy, or LAMMPS APIs), fake mock data instead of real file parsing, "
        "or produces incorrect/broken outputs or fails the core objective, the score MUST BE RULED AROUND 20% (strictly capped between 0 and 25 out of 100). "
        "Do NOT be forgiving. A broken or misleading scientific script ruins months of computational work and must be severely punished.\n"
        "2. STRICT 100-POINT CATEGORY BREAKDOWN:\n"
        "   - Execution & Syntax Integrity (25 pts max): 100% executable without runtime errors, headless plot saving (plt.savefig and plt.close), proper type unpacking.\n"
        "   - Scientific & Domain Correctness (35 pts max): Physical validity, accurate formulas (RDF, density, PBC wrapping, neighbor cutoffs), proper unit conversions (Å, fs, eV, g/cm³), correct stoichiometry.\n"
        "   - Task & Constraint Compliance (25 pts max): Rigorous completion of all prompt requirements without shortcuts, omissions, or dummy placeholders.\n"
        "   - Code Architecture & Best Practices (15 pts max): Vectorization, clear variable naming, modular structure, proper error boundaries.\n"
        "3. EXTREME PREJUDICE: If the code takes shortcuts, uses synthetic dummy data when real trajectory parsing was requested, "
        "or misses edge cases, heavily deduct points. A score above 80 is reserved ONLY for flawless, production-ready, peer-review-grade implementations.\n\n"
        "OUTPUT REQUIREMENT: You must return ONLY valid, parseable JSON conforming to the requested schema with no markdown wrapping."
    )

    @staticmethod
    def extract_python_code(raw_text: str) -> str:
        """Extracts Python code from markdown text or raw string."""
        if not raw_text:
            return ""

        # Strip details / thinking blocks
        cleaned = re.sub(r"<details[\s\S]*?<\/details>", "", raw_text, flags=re.IGNORECASE).strip()

        # Match fenced code blocks
        blocks = re.findall(r"```(?:python|py)?\s*?\n([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
        if blocks:
            candidates = [b.strip() for b in blocks if b.strip()]
            if candidates:
                return max(candidates, key=len)

        # Fallback to any fenced block
        blocks_any = re.findall(r"```[a-zA-Z0-9_-]*\s*?\n([\s\S]*?)```", cleaned)
        if blocks_any:
            return max(blocks_any, key=len).strip()

        # Fallback: lines starting with python indicators
        lines = cleaned.splitlines()
        start_idx = -1
        has_python_statement = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ", "def ", "class ", "if __name__")):
                if start_idx == -1:
                    start_idx = i
                has_python_statement = True
                break

        if has_python_statement and start_idx != -1:
            return "\n".join(lines[start_idx:]).strip()

        return ""

    @staticmethod
    def extract_sandbox_output(raw_text: str) -> str:
        """Extracts Docker sandbox execution status and container output if present."""
        if not raw_text:
            return ""
        
        match = re.search(r"###\s*(?:🐳\s*)?(?:Secure\s*)?Docker Sandbox Execution([\s\S]*?)(?:---|\Z)", raw_text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
        
        match_out = re.search(r"\*\*Container Output\*\*:\s*```[a-zA-Z]*\s*([\s\S]*?)```", raw_text)
        if match_out:
            return match_out.group(1).strip()
            
        return ""

    def _get_api_key(self) -> str:
        key = getattr(settings, "GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
        return key.strip()

    def evaluate(
        self,
        code_or_response: str,
        prompt: Optional[str] = None,
        execution_output: Optional[str] = None,
        session_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs strict evaluation of code using Gemini 3.8 Flash.
        Persists the result into the evaluations table and returns the audit dictionary.
        """
        if not code_or_response or not code_or_response.strip():
            return {
                "score": 0,
                "grade": "F",
                "verdict": "FAIL",
                "summary": "Evaluation failed: Empty code or response provided.",
                "execution_status": "Failed",
                "category_scores": {"execution_integrity": 0, "scientific_accuracy": 0, "task_compliance": 0, "code_quality": 0},
                "critical_issues": ["Empty code submission: no Python script was provided to evaluate."],
                "strengths": [],
                "recommendations": ["Provide a complete, syntactically valid Python script."],
                "passed_checks": [],
                "failed_checks": ["Empty code submission"],
                "model": "none",
                "latency_seconds": 0.0,
                "extracted_code": ""
            }

        code = self.extract_python_code(code_or_response)
        if not code:
            return {
                "score": 0,
                "grade": "F",
                "verdict": "FAIL",
                "summary": "Evaluation failed: Empty Python code block extracted from response.",
                "execution_status": "Failed",
                "category_scores": {"execution_integrity": 0, "scientific_accuracy": 0, "task_compliance": 0, "code_quality": 0},
                "critical_issues": ["No executable Python code found in the model response."],
                "strengths": [],
                "recommendations": ["Ensure the model outputs code inside a ```python ... ``` block."],
                "passed_checks": [],
                "failed_checks": ["No executable Python code found"],
                "model": "none",
                "latency_seconds": 0.0,
                "extracted_code": ""
            }

        api_key = self._get_api_key()
        if not api_key:
            return {
                "score": 0,
                "grade": "F",
                "verdict": "FAIL",
                "summary": "Evaluation failed: GEMINI_API_KEY is not configured in .env.",
                "execution_status": "Failed",
                "category_scores": {"execution_integrity": 0, "scientific_accuracy": 0, "task_compliance": 0, "code_quality": 0},
                "critical_issues": ["Missing GEMINI_API_KEY in environment variables."],
                "strengths": [],
                "recommendations": ["Add a valid GEMINI_API_KEY to your .env file to enable Gemini code evaluation."],
                "passed_checks": [],
                "failed_checks": ["Missing GEMINI_API_KEY in environment variables."],
                "model": "none",
                "latency_seconds": 0.0,
                "extracted_code": code
            }

        if not execution_output:
            execution_output = self.extract_sandbox_output(code_or_response)

        task_desc = prompt.strip() if prompt and prompt.strip() else "Analyze scientific data/materials system with Python script."

        user_content = (
            f"USER PROMPT / TASK REQUIREMENT:\n{task_desc}\n\n"
            f"GENERATED PYTHON SCRIPT:\n```python\n{code}\n```\n\n"
        )
        if execution_output:
            user_content += f"EXECUTION / SANDBOX LOG OUTPUT:\n{execution_output}\n\n"
        else:
            user_content += "EXECUTION / SANDBOX LOG OUTPUT:\nNone provided (analyze static execution integrity).\n\n"

        user_content += (
            "Evaluate this script strictly out of 100 with extreme prejudice. "
            "If the code has errors, invalid logic, hallucinations, fake dummy data, or incorrect output, rule it around 20%.\n\n"
            "Return valid JSON matching this schema:\n"
            "{\n"
            '  "score": <integer 0-100>,\n'
            '  "grade": "<A+ | A | B | C | D | F>",\n'
            '  "verdict": "<PASS | BORDERLINE | FAIL>",\n'
            '  "summary": "<Direct, blunt executive critique with zero sugarcoating>",\n'
            '  "execution_status": "<Success | Failed | Incomplete / Dummy>",\n'
            '  "category_scores": {\n'
            '    "execution_integrity": <score out of 25>,\n'
            '    "scientific_accuracy": <score out of 35>,\n'
            '    "task_compliance": <score out of 25>,\n'
            '    "code_quality": <score out of 15>\n'
            "  },\n"
            '  "critical_issues": [<string>, ...],\n'
            '  "strengths": [<string>, ...],\n'
            '  "recommendations": [<string>, ...]\n'
            "}"
        )

        full_prompt = f"{self.STRICT_SYSTEM_PROMPT}\n\n{user_content}"

        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1,
                "thinkingConfig": {
                    "thinkingBudget": 256
                }
            }
        }

        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }

        start_time = time.time()
        result_data, used_model, error_msg = self._call_gemini_api(payload, headers)
        latency = time.time() - start_time

        if not result_data:
            return {
                "score": 0,
                "grade": "F",
                "verdict": "FAIL",
                "summary": f"Gemini evaluation request failed: {error_msg}",
                "execution_status": "Failed",
                "category_scores": {"execution_integrity": 0, "scientific_accuracy": 0, "task_compliance": 0, "code_quality": 0},
                "critical_issues": [f"API Error: {error_msg}"],
                "strengths": [],
                "recommendations": ["Check network connection and Gemini API rate limits/quotas."],
                "passed_checks": [],
                "failed_checks": [f"API Error: {error_msg}"],
                "model": used_model or self.PRIMARY_MODEL,
                "latency_seconds": round(latency, 2),
                "extracted_code": code
            }

        # Normalize score and grade
        score = int(result_data.get("score", 0))
        grade = str(result_data.get("grade", "F"))
        verdict = str(result_data.get("verdict", "FAIL" if score < 50 else ("BORDERLINE" if score < 80 else "PASS")))
        summary = str(result_data.get("summary", ""))
        exec_status = str(result_data.get("execution_status", "Untested"))
        cat_scores = result_data.get("category_scores", {})
        critical_issues = result_data.get("critical_issues", [])
        strengths = result_data.get("strengths", [])
        recommendations = result_data.get("recommendations", [])

        # Persist into DB
        eval_id = None
        if session_id is not None:
            try:
                eval_report = {
                    "summary": summary,
                    "verdict": verdict,
                    "execution_status": exec_status,
                    "critical_issues": critical_issues,
                    "strengths": strengths,
                    "recommendations": recommendations,
                    "evaluator_model": used_model
                }
                eval_id = insert_evaluation(
                    session_id=session_id,
                    model_name=used_model,
                    prompt=task_desc,
                    generated_code=code,
                    score=float(score),
                    grade=grade,
                    category_scores_json=json.dumps(cat_scores),
                    evaluation_report_json=json.dumps(eval_report),
                    latency_seconds=round(latency, 2)
                )
            except Exception as e:
                print(f"Notice: Failed to insert evaluation into DB: {e}")

        return {
            "evaluation_id": eval_id,
            "score": score,
            "grade": grade,
            "verdict": verdict,
            "summary": summary,
            "execution_status": exec_status,
            "category_scores": cat_scores,
            "critical_issues": critical_issues,
            "strengths": strengths,
            "recommendations": recommendations,
            "passed_checks": strengths,
            "failed_checks": critical_issues,
            "model": used_model,
            "latency_seconds": round(latency, 2),
            "extracted_code": code,
            "evaluation_type": "gemini_strict"
        }

    def _call_gemini_api(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        """Invokes Gemini API with fast retries and seamless model fallbacks."""
        models_to_try = [self.PRIMARY_MODEL] + [m for m in self.FALLBACK_MODELS if m != self.PRIMARY_MODEL]
        last_error = ""

        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            max_attempts = 2 if model == self.PRIMARY_MODEL else 1
            for attempt in range(max_attempts):
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=25)
                    if resp.status_code == 200:
                        body = resp.json()
                        candidates = body.get("candidates", [])
                        if candidates and "content" in candidates[0]:
                            parts = candidates[0]["content"].get("parts", [])
                            if parts and "text" in parts[0]:
                                text = parts[0]["text"].strip()
                                text = re.sub(r"^```(?:json)?\s*", "", text)
                                text = re.sub(r"\s*```$", "", text)
                                parsed = json.loads(text)
                                return parsed, model, None
                        last_error = "Empty or malformed candidate in response"
                    elif resp.status_code == 429:
                        last_error = f"HTTP 429: Rate limit/quota exceeded for {model}"
                        break
                    elif resp.status_code == 503:
                        last_error = f"HTTP 503: Model {model} temporary spike"
                        time.sleep(1.0)
                        continue
                    else:
                        last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                        break
                except Exception as e:
                    last_error = str(e)
                    time.sleep(0.5)

        return None, None, last_error
