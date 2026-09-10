import pytest
import requests
from backend.agents.baseline_agent import SingleShotAgent

def test_generate_response_success(mocker):
    mock_resp = mocker.MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "Cutoff ecutwfc defines the plane-wave basis kinetic energy.",
        "total_duration": 1200000000,
        "eval_count": 42
    }
    mocker.patch("requests.post", return_value=mock_resp)

    agent = SingleShotAgent(model="batiai/gemma4-e2b:q4")
    res = agent.generate_response("What is ecutwfc?")

    assert res["status"] == "success"
    assert "plane-wave" in res["response"]
    assert res["model"] == "batiai/gemma4-e2b:q4"
    assert res["mode"] == "single_shot"
    assert res["eval_count"] == 42
    assert "latency_seconds" in res

def test_generate_response_connection_error(mocker):
    mocker.patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused"))

    agent = SingleShotAgent(model="batiai/gemma4-e2b:q4")
    res = agent.generate_response("test prompt")

    assert res["status"] == "error"
    assert "Could not connect to Ollama" in res["error"]

def test_generate_response_generic_error(mocker):
    mocker.patch("requests.post", side_effect=Exception("Timeout occurred"))

    agent = SingleShotAgent(model="batiai/gemma4-e2b:q4")
    res = agent.generate_response("test prompt")

    assert res["status"] == "error"
    assert "Timeout occurred" in res["error"]

def test_chat_new_session(mocker):
    mock_insert_session = mocker.patch("backend.agents.baseline_agent.insert_session", return_value=101)
    mock_insert_message = mocker.patch("backend.agents.baseline_agent.insert_message")
    
    mock_agent_res = {
        "status": "success",
        "response": "Here is the calculation result.",
        "model": "batiai/gemma4-e2b:q4",
        "latency_seconds": 0.85
    }
    mocker.patch.object(SingleShotAgent, "generate_response", return_value=mock_agent_res)

    agent = SingleShotAgent()
    res = agent.chat("Explain RDF calculation")

    assert res["status"] == "success"
    assert res["session_id"] == 101
    assert mock_insert_session.called
    assert mock_insert_message.call_count == 2  # user and assistant

def test_chat_existing_session(mocker):
    mock_insert_session = mocker.patch("backend.agents.baseline_agent.insert_session")
    mock_insert_message = mocker.patch("backend.agents.baseline_agent.insert_message")
    
    mock_agent_res = {
        "status": "success",
        "response": "Response for existing session.",
        "model": "batiai/gemma4-e2b:q4",
        "latency_seconds": 0.5
    }
    mocker.patch.object(SingleShotAgent, "generate_response", return_value=mock_agent_res)

    agent = SingleShotAgent()
    res = agent.chat("Follow-up question", session_id=42)

    assert res["session_id"] == 42
    assert not mock_insert_session.called
    assert mock_insert_message.call_count == 2

def test_chat_failed_generation(mocker):
    mocker.patch("backend.agents.baseline_agent.insert_session", return_value=102)
    mock_insert_message = mocker.patch("backend.agents.baseline_agent.insert_message")
    
    mock_agent_res = {
        "status": "error",
        "error": "Ollama offline",
        "model": "batiai/gemma4-e2b:q4",
        "latency_seconds": 0.1
    }
    mocker.patch.object(SingleShotAgent, "generate_response", return_value=mock_agent_res)

    agent = SingleShotAgent()
    res = agent.chat("Question that fails")

    assert res["status"] == "error"
    assert mock_insert_message.call_count == 1  # Only user message logged
