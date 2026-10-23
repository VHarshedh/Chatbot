import re
from typing import Dict, Any, List, Optional
from .gemini_evaluator import GeminiEvaluator

class ScriptEvaluator:
    """
    Scientific Script Evaluation Engine powered by the configured Gemini model.
    Combines local syntax checks with schema-validated scientific review.
    """

    def __init__(self):
        self._gemini_evaluator = GeminiEvaluator()

    @staticmethod
    def extract_python_code(raw_text: str) -> str:
        """Extracts Python code blocks from markdown-formatted LLM responses."""
        return GeminiEvaluator.extract_python_code(raw_text)

    @staticmethod
    def extract_sandbox_output(raw_text: str) -> str:
        """Extracts container logs from Docker execution sections."""
        return GeminiEvaluator.extract_sandbox_output(raw_text)

    @staticmethod
    def extract_metrics(text: str) -> Dict[str, float]:
        """Parses numerical metrics from text."""
        metrics = {}
        for line in (text or "").splitlines():
            m = re.findall(r"([a-zA-Z0-9_]+)\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", line)
            for k, v in m:
                try:
                    metrics[k] = float(v)
                except ValueError:
                    pass
        return metrics

    def evaluate(
        self,
        raw_response: str,
        prompt: Optional[str] = None,
        notebook_context: Optional[Dict[str, Any]] = None,
        execution_output: Optional[str] = None,
        session_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a zero-shot or agentic generated script using the configured Gemini model.
        Blocking code defects cap the score at 25; service failures remain ungraded.
        """
        code = self.extract_python_code(raw_response)
        if not execution_output:
            execution_output = self.extract_sandbox_output(raw_response)

        result = self._gemini_evaluator.evaluate(
            code_or_response=code if code else raw_response,
            prompt=prompt,
            execution_output=execution_output,
            session_id=session_id,
            notebook_context=notebook_context
        )
        result["notebook_context"] = notebook_context
        return result
