import json

import pytest
from fastapi.testclient import TestClient

from backend.evaluation.gemini_evaluator import GeminiEvaluator
from backend.main import app


@pytest.fixture
def audit(mocker):
    evaluator = GeminiEvaluator()
    mocker.patch.object(evaluator, "_get_api_key", return_value="test-key")
    persist = mocker.patch("backend.evaluation.gemini_evaluator.insert_evaluation", return_value=7)
    report = {"category_scores": dict(evaluator.CATEGORY_LIMITS), "summary": "Code reviewed.",
              "execution_status": "Untested", "blocking_failure": False,
              "critical_issues": [], "strengths": ["Simple implementation"], "recommendations": []}
    call = mocker.patch.object(evaluator, "_call_gemini_api", return_value=(report, "test-model", None))
    return evaluator, report, call, persist


def test_scores_are_derived_and_execution_not_invented(audit):
    evaluator, report, call, persist = audit
    report.update(score=-40, grade="F", verdict="FAIL")
    result = evaluator.evaluate("x = 1\nprint(x)", execution_output="Success!", notebook_context={"traj": "available"})
    assert result["score"] == 100
    assert result["grade"] == "A+"
    assert result["execution_status"] == "Untested"
    assert result["execution_evidence"] == "supplied_unverified"
    payload = call.call_args.args[0]
    assert "systemInstruction" in payload
    assert "responseJsonSchema" in payload["generationConfig"]
    assert json.loads(payload["contents"][0]["parts"][0]["text"])["notebook_context"] == {"traj": "available"}
    persist.assert_called_once()


@pytest.mark.parametrize("bad", [None, [], {"execution_integrity": 26},
    {"execution_integrity": True, "scientific_accuracy": 35, "task_compliance": 25, "code_quality": 15}])
def test_invalid_categories_not_persisted(audit, bad):
    evaluator, report, _, persist = audit
    report["category_scores"] = bad
    result = evaluator.evaluate("print(1)")
    assert result["audit_status"] == "error"
    assert result["score"] is None
    persist.assert_not_called()


def test_syntax_failure_overrides_glowing_review(audit):
    evaluator, _, _, _ = audit
    result = evaluator.evaluate("```python\ndef broken(:\n    pass\n```")
    assert result["score"] == 25
    assert result["score_cap"] == 25
    assert result["raw_score"] == 75
    assert result["execution_status"] == "Failed"
    assert "compilation failed" in result["critical_issues"][0]


def test_core_failure_caps_score(audit):
    evaluator, report, _, _ = audit
    report.update(blocking_failure=True, critical_issues=["Uses random data instead of required file."])
    result = evaluator.evaluate("print(1)")
    assert result["score"] == 25
    assert result["verdict"] == "FAIL"


def test_api_failure_not_a_code_grade(audit):
    evaluator, _, call, persist = audit
    call.return_value = (None, None, "Quota exhausted")
    result = evaluator.evaluate("print(1)")
    assert result["verdict"] == "ERROR"
    assert result["score"] is None
    persist.assert_not_called()


def test_missing_key_not_persisted(audit, mocker):
    evaluator, _, call, persist = audit
    mocker.patch.object(evaluator, "_get_api_key", return_value="")
    assert evaluator.evaluate("print(1)")["audit_status"] == "error"
    call.assert_not_called()
    persist.assert_not_called()


def test_extraction_preserves_assignments_and_multiple_blocks():
    extract = GeminiEvaluator.extract_python_code
    assert extract("x = 3\nimport math\nprint(math.sqrt(x))").startswith("x = 3")
    assert extract("```python\nx = 3\n```\n```python\nprint(x)\n```") == "x = 3\n\nprint(x)"
    assert extract("```bash\necho hello\n```") == ""


def test_multipart_response_ignores_thoughts(audit, mocker):
    evaluator, report, _, _ = audit
    mocker.patch("backend.evaluation.gemini_evaluator.requests.post", return_value=mocker.Mock(
        status_code=200, json=lambda: {"candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"thought": True, "text": "reasoning"}, {"text": json.dumps(report)[:20]},
            {"text": json.dumps(report)[20:]}
        ]}}]}))
    parsed, model, error = GeminiEvaluator._call_gemini_api(evaluator, {}, {})
    assert parsed == report
    assert error is None


def test_api_endpoint_returns_error_without_fake_report(mocker):
    mocker.patch("backend.main.GeminiEvaluator.evaluate", return_value=GeminiEvaluator._error("Quota exhausted"))
    response = TestClient(app).post("/api/evaluate/gemini", json={"code": "print(1)"})
    assert response.status_code == 502
    assert response.json()["detail"] == "Quota exhausted"


@pytest.mark.parametrize("streaming", [False, True])
def test_benchmark_audit_error_is_not_saved(mocker, streaming):
    mocker.patch("backend.main.SingleShotAgent.chat", return_value={
        "status": "success", "response": "```python\nprint(4)\n```", "session_id": 37})
    mocker.patch("backend.main.SingleShotAgent.chat_stream", return_value=iter([
        'data: ' + json.dumps({"type": "done", "full_response": "```python\nprint(4)\n```", "session_id": 37}) + '\n\n']))
    mocker.patch("backend.main.ScriptEvaluator.evaluate", return_value=GeminiEvaluator._error("Quota exhausted"))
    persist = mocker.patch("backend.main.insert_evaluation")
    memory = mocker.patch("backend.main._index_evaluated_memory")
    endpoint = "/api/evaluate_zero_shot" + ("/stream" if streaming else "")
    response = TestClient(app).post(endpoint, json={"prompt": "Print 4", "include_notebook_context": False})
    assert response.status_code == (200 if streaming else 502)
    assert "Quota exhausted" in response.text
    assert "eval_complete" not in response.text
    persist.assert_not_called()
    memory.assert_not_called()


def test_truncated_gemini_response_is_rejected(audit, mocker):
    evaluator, report, _, _ = audit
    mocker.patch.object(evaluator, "FALLBACK_MODELS", [])
    mocker.patch("backend.evaluation.gemini_evaluator.requests.post", return_value=mocker.Mock(
        status_code=200, json=lambda: {"candidates": [{"finishReason": "MAX_TOKENS",
            "content": {"parts": [{"text": json.dumps(report)}]}}]}))
    parsed, _, error = GeminiEvaluator._call_gemini_api(evaluator, {}, {})
    assert parsed is None
    assert error
