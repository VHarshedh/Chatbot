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

def test_generate_response_streaming_chunks(mocker):
    mock_resp = mocker.MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = [
        b'{"message": {"thinking": "analyzing request"}, "done": false}',
        b'',  # empty line
        b'invalid json',
        b'{"message": {"content": "Here is the code."}, "eval_count": 25, "total_duration": 500000, "done": true}'
    ]
    mocker.patch("requests.post", return_value=mock_resp)

    agent = SingleShotAgent(model="batiai/gemma4-e2b:q4")
    res = agent.generate_response("Write code")
    assert res["status"] == "success"
    assert "Thinking Process" in res["response"]
    assert "Here is the code." in res["response"]
    assert res["eval_count"] == 25

def test_generate_response_only_thinking_or_empty(mocker):
    mock_resp = mocker.MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = [
        b'{"message": {"thinking": "Just thinking"}, "done": true}'
    ]
    mocker.patch("requests.post", return_value=mock_resp)

    agent = SingleShotAgent(model="batiai/gemma4-e2b:q4")
    res = agent.generate_response("Deep question")
    assert res["status"] == "success"
    assert "Just thinking" in res["response"]

    # Empty response
    mock_resp.iter_lines.return_value = []
    mock_resp.json.return_value = {}
    res_empty = agent.generate_response("Nothing")
    assert res_empty["status"] == "success"
    assert "No response generated" in res_empty["response"]

def test_chat_stream_success(mocker):
    mocker.patch("backend.agents.baseline_agent.insert_session", return_value=501)
    mock_insert_msg = mocker.patch("backend.agents.baseline_agent.insert_message")

    mock_resp = mocker.MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = [
        b'{"message": {"thinking": "analyzing"}, "done": false}',
        b'{"message": {"content": "Answer here"}, "eval_count": 12, "total_duration": 200, "done": true}'
    ]
    mocker.patch("requests.post", return_value=mock_resp)

    agent = SingleShotAgent(model="batiai/gemma4-e2b:q4")
    chunks = list(agent.chat_stream("Hello live stream"))
    assert len(chunks) >= 3
    joined = "".join(chunks)
    assert "session" in joined
    assert "thinking" in joined
    assert "content" in joined
    assert "done" in joined
    assert mock_insert_msg.call_count == 2

def test_chat_stream_connection_error(mocker):
    mocker.patch("backend.agents.baseline_agent.insert_session", return_value=502)
    mocker.patch("backend.agents.baseline_agent.insert_message")
    mocker.patch("requests.post", side_effect=requests.exceptions.ConnectionError("Down"))

    agent = SingleShotAgent()
    chunks = list(agent.chat_stream("Hello fail"))
    joined = "".join(chunks)
    assert "error" in joined
    assert "Could not connect to Ollama" in joined

def test_generate_response_with_docker_execution_molecular(mocker):
    # Mock LLM returning Python code for molecular extxyz
    mock_resp = mocker.MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "Here is the parsing script:\n```python\nfrom ase.io import read\ntraj = read('/mnt/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz')\nprint('Total Energy:', -185982.25)\n```",
        "total_duration": 1000000,
        "eval_count": 50
    }
    mocker.patch("requests.post", return_value=mock_resp)
    
    mock_sandbox = mocker.patch("backend.tools.sandbox_tools.run_python_in_sandbox", return_value={
        "exit_code": 0,
        "stdout": "Total Energy: -185982.25\n",
        "stderr": "",
        "artifacts": []
    })
    mock_attach = mocker.patch("backend.notebooks.attach_to_notebook.attach_script_to_notebook")

    agent = SingleShotAgent()
    res = agent.generate_response("Calculate energy of anatase tio2 water molecular extxyz", run_sandbox=True)

    assert res["status"] == "success"
    assert mock_sandbox.called
    assert "Secure Docker Sandbox Execution" in res["response"]
    assert "Total Energy: -185982.25" in res["response"]
    assert "Success (Exit Code 0)" in res["response"]
    # Molecular extxyz should NOT attach to notebook
    assert not mock_attach.called

def test_generate_response_default_zero_shot_isolation(mocker):
    # Verify default run_sandbox=False does NOT trigger sandbox execution
    mock_resp = mocker.MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "Here is the code:\n```python\nimport numpy as np\nprint('Computed RDF')\n```",
        "total_duration": 1000000,
        "eval_count": 40
    }
    mocker.patch("requests.post", return_value=mock_resp)
    mock_sandbox = mocker.patch("backend.tools.sandbox_tools.run_python_in_sandbox")
    mock_attach = mocker.patch("backend.notebooks.attach_to_notebook.attach_script_to_notebook")

    agent = SingleShotAgent()
    res = agent.generate_response("Compute RDF for anatase tio2")

    assert res["status"] == "success"
    assert not mock_sandbox.called
    assert not mock_attach.called
    assert "Secure Docker Sandbox Execution" not in res["response"]

def test_generate_response_with_docker_execution_standard(mocker):
    # Mock LLM returning Python code for anatase tio2 alone
    mock_resp = mocker.MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "Here is the code:\n```python\nimport numpy as np\nprint('Computed RDF')\nprint('Done')\n```",
        "total_duration": 1000000,
        "eval_count": 40
    }
    mocker.patch("requests.post", return_value=mock_resp)
    
    mock_sandbox = mocker.patch("backend.tools.sandbox_tools.run_python_in_sandbox", return_value={
        "exit_code": 0,
        "stdout": "Computed RDF\nDone\n",
        "stderr": "",
        "artifacts": ["plot.png"]
    })
    mock_attach = mocker.patch("backend.notebooks.attach_to_notebook.attach_script_to_notebook", return_value={
        "target_cell_index": 6
    })

    agent = SingleShotAgent()
    res = agent.generate_response("Compute RDF for anatase tio2", run_sandbox=True)

    assert res["status"] == "success"
    assert mock_sandbox.called
    assert "plot.png" in res["response"]
    assert "Secure Docker Sandbox Execution" in res["response"]
    # Standard anatase tio2 should attach to notebook
    assert mock_attach.called
    assert "Colab Notebook Ready" in res["response"]

def test_generate_response_with_history(mocker):
    captured_payload = {}
    def mock_post(url, json=None, **kwargs):
        nonlocal captured_payload
        captured_payload = json
        mock_resp = mocker.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": "Anatase bandgap is 3.2 eV.",
            "total_duration": 100,
            "eval_count": 10
        }
        return mock_resp

    mocker.patch("requests.post", side_effect=mock_post)

    history = [
        {"role": "user", "content": "What is Anatase TiO2?"},
        {"role": "assistant", "content": "Anatase is a metastable mineral form of titanium dioxide."}
    ]

    agent = SingleShotAgent()
    res = agent.generate_response("What is its bandgap?", history=history)

    assert res["status"] == "success"
    assert "messages" in captured_payload
    msgs = captured_payload["messages"]
    # system + 2 history messages + current user message = 4
    assert len(msgs) == 4
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "What is Anatase TiO2?"
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] == "Anatase is a metastable mineral form of titanium dioxide."
    assert msgs[3]["role"] == "user"
    assert msgs[3]["content"] == "What is its bandgap?"

def test_chat_with_existing_history(mocker):
    fake_history = [
        {"role": "user", "content": "Tell me about Pt."},
        {"role": "assistant", "content": "Platinum is a transition metal."}
    ]
    mock_get_history = mocker.patch("backend.agents.baseline_agent.get_chat_history", return_value=fake_history)
    mocker.patch("backend.agents.baseline_agent.insert_message")
    mock_gen = mocker.patch.object(SingleShotAgent, "generate_response", return_value={
        "status": "success",
        "response": "Its lattice constant is 3.92 Angstroms."
    })

    agent = SingleShotAgent()
    res = agent.chat("What is its lattice constant?", session_id=55)

    assert res["status"] == "success"
    assert mock_get_history.called
    assert mock_get_history.call_args[1]["session_id"] == 55
    assert mock_gen.called
    assert mock_gen.call_args[1]["history"] == fake_history

def test_chat_stream_with_history(mocker):
    fake_history = [
        {"role": "user", "content": "Turn 1 question"},
        {"role": "assistant", "content": "Turn 1 answer"}
    ]
    mocker.patch("backend.agents.baseline_agent.get_chat_history", return_value=fake_history)
    mocker.patch("backend.agents.baseline_agent.insert_message")

    captured_payload = {}
    def mock_post(url, json=None, **kwargs):
        nonlocal captured_payload
        captured_payload = json
        mock_resp = mocker.MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = [
            json.dumps({"message": {"content": "Turn 2 response"}}).encode("utf-8")
        ]
        return mock_resp

    mocker.patch("requests.post", side_effect=mock_post)

    agent = SingleShotAgent()
    events = list(agent.chat_stream("Turn 2 question", session_id=77))

    assert len(events) > 0
    msgs = captured_payload["messages"]
    assert len(msgs) == 4
    assert msgs[1]["content"] == "Turn 1 question"
    assert msgs[2]["content"] == "Turn 1 answer"
    assert msgs[3]["content"] == "Turn 2 question"

