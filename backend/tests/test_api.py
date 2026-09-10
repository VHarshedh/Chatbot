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
    assert len(prompts) == 4
    domains = [p["domain"] for p in prompts]
    assert "DFT" in domains
    assert "Molecular Dynamics" in domains

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

@pytest.mark.asyncio
async def test_startup_event():
    from backend.main import startup_event
    await startup_event()
