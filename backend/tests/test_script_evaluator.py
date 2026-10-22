import pytest
from unittest.mock import patch, MagicMock
from backend.evaluation.script_evaluator import ScriptEvaluator
from backend.evaluation.gemini_evaluator import GeminiEvaluator


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


def test_extract_sandbox_output():
    evaluator = ScriptEvaluator()
    msg = """
### Docker Sandbox Execution
**Exit Code:** `0`
```text
Calculated mean residence time: 0.298 ns
Successfully generated figure.
```
"""
    extracted = evaluator.extract_sandbox_output(msg)
    assert "Calculated mean residence time: 0.298 ns" in extracted


def test_gemini_evaluator_empty_code():
    evaluator = GeminiEvaluator()
    res = evaluator.evaluate("")
    assert res["score"] == 0
    assert res["verdict"] == "FAIL"
    assert res["grade"] == "F"
    assert "empty" in res["summary"].lower()
    assert len(res["critical_issues"]) >= 1


def test_gemini_evaluator_mock_success():
    evaluator = GeminiEvaluator()

    mock_gemini_response = {
        "score": 18,
        "grade": "F",
        "verdict": "FAIL",
        "summary": "The script uses fabricated dummy data and failed to load real trajectory files.",
        "execution_status": "Failed",
        "category_scores": {
            "execution_integrity": 5,
            "scientific_accuracy": 5,
            "task_compliance": 5,
            "code_quality": 3
        },
        "critical_issues": [
            "Fake mock data: Simulated random coordinates instead of reading trajectory.",
            "Missing neighbor analysis logic."
        ],
        "strengths": [
            "Clean headless matplotlib usage."
        ],
        "recommendations": [
            "Parse real .xyz / .dump files using ASE read."
        ]
    }

    with patch.object(evaluator, "_call_gemini_api", return_value=(mock_gemini_response, "gemini-3.5-flash-lite", None)):
        res = evaluator.evaluate(
            code_or_response="```python\nimport numpy as np\nx = np.random.rand(10, 3)\n```",
            prompt="Analyze water trajectory",
            execution_output="Process completed with synthetic points."
        )

        assert res["score"] == 18
        assert res["grade"] == "F"
        assert res["verdict"] == "FAIL"
        assert "Fake mock data" in res["critical_issues"][0]
        assert res["model"] == "gemini-3.5-flash-lite"
        assert res["category_scores"]["execution_integrity"] == 5


def test_script_evaluator_delegation():
    evaluator = ScriptEvaluator()

    mock_gemini_response = {
        "score": 92,
        "grade": "A",
        "verdict": "PASS",
        "summary": "Flawless scientific implementation with robust PBC wrapping.",
        "execution_status": "Verified",
        "category_scores": {
            "execution_integrity": 25,
            "scientific_accuracy": 33,
            "task_compliance": 22,
            "code_quality": 12
        },
        "critical_issues": [],
        "strengths": ["Excellent vectorized PBC handling"],
        "recommendations": []
    }

    with patch.object(evaluator._gemini_evaluator, "_call_gemini_api", return_value=(mock_gemini_response, "gemini-3.5-flash-lite", None)):
        res = evaluator.evaluate(
            raw_response="```python\nimport numpy as np\nprint('Pass')\n```",
            prompt="Run full analysis"
        )
        assert res["score"] == 92
        assert res["grade"] == "A"
        assert res["verdict"] == "PASS"
