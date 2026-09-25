import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200

def test_get_model_info():
    response = client.get("/api/model_info")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "single_shot"
    assert data["tools_enabled"] is False
    assert data["rag_enabled"] is False

def test_get_benchmark_prompts():
    response = client.get("/api/benchmark_prompts")
    assert response.status_code == 200
    prompts = response.json()
    assert len(prompts) == 5
    domains = [p["domain"] for p in prompts]
    assert "DFT" in domains
    assert "Molecular Dynamics" in domains
    assert "Interfacial Script Evaluation" in domains

def test_chat_endpoint_success(mocker):
    mock_chat_res = {
        "status": "success",
        "response": "Platinum has an fcc structure.",
        "model": "batiai/gemma4-e2b:q4",
        "latency_seconds": 0.42,
        "session_id": 99
    }
    mocker.patch("backend.main.SingleShotAgent.chat", return_value=mock_chat_res)

    payload = {"message": "Describe Pt crystal structure"}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "Platinum" in data["response"]
    assert data["session_id"] == 99

def test_chat_endpoint_empty_message():
    response = client.post("/api/chat", json={"message": "   "})
    assert response.status_code == 400
    assert "Message cannot be empty" in response.json()["detail"]

def test_chat_stream_endpoint_success(mocker):
    from backend.agents.baseline_agent import SingleShotAgent
    def fake_stream(*args, **kwargs):
        yield 'data: {"type": "session", "session_id": 10}\n\n'
        yield 'data: {"type": "content", "delta": "Hello"}\n\n'
        yield 'data: {"type": "done", "latency_seconds": 0.5}\n\n'

    mocker.patch.object(SingleShotAgent, "chat_stream", side_effect=fake_stream)

    response = client.post("/api/chat/stream", json={"message": "Hello stream"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    body = response.text
    assert "Hello" in body
    assert "session" in body

def test_chat_stream_endpoint_empty_message():
    response = client.post("/api/chat/stream", json={"message": ""})
    assert response.status_code == 400
    assert "Message cannot be empty" in response.json()["detail"]

def test_sessions_crud(mocker):
    # Mock database queries
    mocker.patch(
        "backend.main.execute_query",
        side_effect=[
            [{"id": 1, "title": "Test Session", "created_at": "2026-09-10", "last_message_at": "2026-09-10"}], # list
            [{"id": 1, "role": "user", "content": "hello", "timestamp": "2026-09-10", "agent_used": None}],     # get messages
            [], # delete messages
            []  # delete session
        ]
    )
    mocker.patch("backend.main.insert_session", return_value=12)

    # 1. Create session
    create_res = client.post("/api/sessions", json={"title": "Custom Title"})
    assert create_res.status_code == 200
    assert create_res.json()["session_id"] == 12

    # 2. List sessions
    list_res = client.get("/api/sessions")
    assert list_res.status_code == 200
    assert len(list_res.json()) == 1

    # 3. Get session messages
    msg_res = client.get("/api/sessions/1/messages")
    assert msg_res.status_code == 200
    assert len(msg_res.json()) == 1
    assert msg_res.json()[0]["content"] == "hello"

    # 4. Delete session
    del_res = client.delete("/api/sessions/1")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

def test_evaluate_zero_shot_success(mocker):
    mock_chat_res = {
        "status": "success",
        "response": "```python\nimport numpy as np\nimport ase\n# m_sur cus\n```",
        "model": "batiai/gemma4-e2b:q4",
        "latency_seconds": 1.2,
        "session_id": 50
    }
    mocker.patch("backend.main.SingleShotAgent.chat", return_value=mock_chat_res)
    mocker.patch("backend.main.insert_evaluation", return_value=1)

    response = client.post("/api/evaluate_zero_shot", json={"session_id": 50})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["evaluation_id"] == 1
    assert "score" in data
    assert "grade" in data
    assert "category_scores" in data

def test_evaluate_zero_shot_failure(mocker):
    mock_chat_res = {
        "status": "error",
        "error": "Ollama timeout",
        "model": "batiai/gemma4-e2b:q4"
    }
    mocker.patch("backend.main.SingleShotAgent.chat", return_value=mock_chat_res)

    response = client.post("/api/evaluate_zero_shot", json={})
    assert response.status_code == 500
    assert "Ollama timeout" in response.json()["detail"]

def test_list_evaluations(mocker):
    mocker.patch("backend.main.get_evaluations", return_value=[{"id": 1, "score": 85.0}])
    response = client.get("/api/evaluations")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["score"] == 85.0

def test_read_root_fallback(mocker):
    mocker.patch("backend.main.FRONTEND_DIR", mocker.MagicMock(exists=lambda: False, __truediv__=lambda self, x: mocker.MagicMock(exists=lambda: False)))
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_startup_event():
    import asyncio
    from backend.main import startup_event
    asyncio.run(startup_event())

def test_evaluate_zero_shot_stream_success(mocker):
    mock_events = [
        'data: {"type": "session", "session_id": 99}\n\n',
        'data: {"type": "thinking", "delta": "reasoning"}\n\n',
        'data: {"type": "content", "delta": "```python\\nimport numpy as np\\nimport ase\\n```"}\n\n',
        'data: {"type": "done", "latency_seconds": 1.5, "session_id": 99, "full_response": "```python\\nimport numpy as np\\nimport ase\\n```"}\n\n'
    ]
    mocker.patch("backend.main.SingleShotAgent.chat_stream", return_value=iter(mock_events))
    mocker.patch("backend.main.insert_evaluation", return_value=12)

    response = client.post("/api/evaluate_zero_shot/stream", json={"session_id": 99})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    text = response.text
    assert "eval_complete" in text
    assert '"score"' in text
    assert '"evaluation_id": 12' in text

def test_chat_endpoint_agentic_mode(mocker):
    mock_agentic_res = {
        "status": "success",
        "response": "Agentic RAG answer with sandbox output",
        "model": "batiai/gemma4-e2b:q4",
        "latency_seconds": 1.1,
        "session_id": 105,
        "agent_used": "AgenticRAG",
        "mode": "agentic"
    }
    mocker.patch("backend.main.AgenticRAGAgent.chat", return_value=mock_agentic_res)

    payload = {"message": "Run density analysis", "mode": "agentic"}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["agent_used"] == "AgenticRAG"
    assert data["mode"] == "agentic"

def test_chat_stream_endpoint_agentic_mode(mocker):
    def fake_agentic_stream(*args, **kwargs):
        yield 'data: {"type": "session", "session_id": 106, "mode": "agentic"}\n\n'
        yield 'data: {"type": "agent_step", "step": "rag_start", "title": "RAG", "status": "running"}\n\n'
        yield 'data: {"type": "content", "delta": "Agentic content"}\n\n'
        yield 'data: {"type": "done", "latency_seconds": 1.2, "mode": "agentic"}\n\n'

    mocker.patch("backend.main.AgenticRAGAgent.chat_stream", side_effect=fake_agentic_stream)

    payload = {"message": "Stream agentic RAG", "mode": "agentic"}
    response = client.post("/api/chat/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    text = response.text
    assert "agent_step" in text
    assert "Agentic content" in text

def test_prompt_crud_and_reset_flow():
    # 1. Reset first to ensure clean state
    reset_res = client.post("/api/benchmark_prompts/reset")
    assert reset_res.status_code == 200
    assert len(reset_res.json()["prompts"]) == 5

    # 2. Create a new custom prompt
    create_payload = {
        "id": "custom_test_prompt",
        "title": "Custom Test Prompt",
        "domain": "Machine Learning",
        "prompt": "Test generating an ML potential training script."
    }
    create_res = client.post("/api/benchmark_prompts", json=create_payload)
    assert create_res.status_code == 200
    created = create_res.json()
    assert created["id"] == "custom_test_prompt"
    assert created["title"] == "Custom Test Prompt"
    assert created["domain"] == "Machine Learning"

    # 3. Retrieve the created prompt
    get_res = client.get("/api/benchmark_prompts/custom_test_prompt")
    assert get_res.status_code == 200
    assert get_res.json()["title"] == "Custom Test Prompt"

    # 4. Update the prompt
    update_payload = {
        "title": "Updated Custom Test Prompt",
        "domain": "Neural Potential",
        "prompt": "Updated prompt text for testing."
    }
    update_res = client.put("/api/benchmark_prompts/custom_test_prompt", json=update_payload)
    assert update_res.status_code == 200
    updated = update_res.json()
    assert updated["title"] == "Updated Custom Test Prompt"
    assert updated["domain"] == "Neural Potential"
    assert updated["prompt"] == "Updated prompt text for testing."

    # 5. Delete the prompt
    del_res = client.delete("/api/benchmark_prompts/custom_test_prompt")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "success"

    # 6. Verify prompt is deleted (404)
    get_del_res = client.get("/api/benchmark_prompts/custom_test_prompt")
    assert get_del_res.status_code == 404

    # 7. Delete nonexistent prompt returns 404
    del_404_res = client.delete("/api/benchmark_prompts/non_existent_id")
    assert del_404_res.status_code == 404

    # 8. Reset restores original defaults
    reset_again = client.post("/api/benchmark_prompts/reset")
    assert reset_again.status_code == 200
    prompts = client.get("/api/benchmark_prompts").json()
    assert len(prompts) == 5


