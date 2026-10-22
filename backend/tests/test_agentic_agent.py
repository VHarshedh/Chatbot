import json
import pytest
from unittest.mock import MagicMock
from backend.agents.agentic_agent import AgenticRAGAgent

def test_agentic_agent_init(mocker):
    mocker.patch("backend.agents.agentic_agent.HybridRetriever")
    agent = AgenticRAGAgent(model="custom-model", base_url="http://mock-ollama:11434/")
    assert agent.model == "custom-model"
    assert agent.base_url == "http://mock-ollama:11434"

def test_is_execution_request():
    agent = AgenticRAGAgent.__new__(AgenticRAGAgent)
    assert agent._is_execution_request("Please run python script to calculate density profile") is True
    assert agent._is_execution_request("Simulate interfacial water with deepmd model") is True
    assert agent._is_execution_request("Plot the radial distribution function rdf") is True
    assert agent._is_execution_request("What is the coordination number of Ir in IrO2?") is False
    assert agent._is_execution_request("Explain the mathematical definition of survival probability P(t) and radial distribution function") is False
    assert agent._is_execution_request("What is the survival probability of water on TiO2?") is False
    assert agent._is_execution_request("Define radial distribution function") is False

def test_chat_stream_non_execution(mocker):
    mock_retriever = mocker.patch("backend.agents.agentic_agent.HybridRetriever")
    mock_retriever.return_value.retrieve.return_value = "Retrieved domain context about water."
    mocker.patch("backend.agents.agentic_agent.insert_session", return_value=42)
    mocker.patch("backend.agents.agentic_agent.insert_message")

    # Mock streaming response
    mock_stream_response = MagicMock()
    mock_stream_response.iter_lines.return_value = [
        json.dumps({"message": {"thinking": "Analyzing question..."}}).encode("utf-8"),
        json.dumps({"message": {"content": "Water is H2O."}}).encode("utf-8"),
        json.dumps({"eval_count": 25}).encode("utf-8")
    ]
    mocker.patch.object(AgenticRAGAgent, "_call_llm", return_value=mock_stream_response)

    agent = AgenticRAGAgent()
    events = list(agent.chat_stream("Explain water chemistry"))

    parsed = [json.loads(e.replace("data: ", "").strip()) for e in events if e.strip()]
    event_types = [p["type"] for p in parsed]

    assert "session" in event_types
    assert "agent_step" in event_types
    assert "thinking" in event_types
    assert "content" in event_types
    assert "done" in event_types

    done_event = next(p for p in parsed if p["type"] == "done")
    assert done_event["mode"] == "agentic"
    assert "Water is H2O." in done_event["full_response"]

def test_chat_stream_execution_success(mocker):
    mock_retriever = mocker.patch("backend.agents.agentic_agent.HybridRetriever")
    mock_retriever.return_value.retrieve.return_value = "Context on RDF calculation"
    mocker.patch("backend.agents.agentic_agent.insert_session", return_value=43)
    mocker.patch("backend.agents.agentic_agent.insert_message")

    # Mock code generation LLM call and synthesis stream
    mock_codegen_res = MagicMock()
    mock_codegen_res.json.return_value = {
        "message": {"content": "```python\nimport numpy as np\nprint('Computed RDF')\n```"}
    }
    
    mock_stream_res = MagicMock()
    mock_stream_res.iter_lines.return_value = [
        json.dumps({"message": {"content": "The simulation analysis succeeded."}}).encode("utf-8")
    ]

    mocker.patch.object(AgenticRAGAgent, "_call_llm", side_effect=[mock_codegen_res, mock_stream_res])
    mocker.patch("backend.agents.agentic_agent.run_python_in_sandbox", return_value={
        "exit_code": 0,
        "stdout": "Computed RDF successfully",
        "stderr": "",
        "artifacts": ["rdf.png"]
    })

    agent = AgenticRAGAgent()
    events = list(agent.chat_stream("Run python script to compute radial distribution"))

    parsed = [json.loads(e.replace("data: ", "").strip()) for e in events if e.strip()]
    steps = [p.get("step") for p in parsed if p["type"] == "agent_step"]

    assert "rag_start" in steps
    assert "rag_complete" in steps
    assert "code_gen_start" in steps
    assert "code_gen_complete" in steps
    assert "sandbox_start" in steps
    assert "sandbox_complete" in steps
    assert "synthesis_start" in steps

def test_chat_stream_execution_failure_and_web_correction(mocker):
    mock_retriever = mocker.patch("backend.agents.agentic_agent.HybridRetriever")
    mock_retriever.return_value.retrieve.return_value = "Context info"
    mocker.patch("backend.agents.agentic_agent.insert_session", return_value=44)
    mocker.patch("backend.agents.agentic_agent.insert_message")

    mock_codegen_res = MagicMock()
    mock_codegen_res.json.return_value = {
        "message": {"content": "```python\nbad_code()\n```"}
    }
    mock_fix_res = MagicMock()
    mock_fix_res.json.return_value = {
        "message": {"content": "```python\nfixed_code()\n```"}
    }
    mock_stream_res = MagicMock()
    mock_stream_res.iter_lines.return_value = [
        json.dumps({"message": {"content": "Fixed code completed."}}).encode("utf-8")
    ]

    mocker.patch.object(AgenticRAGAgent, "_call_llm", side_effect=[mock_codegen_res, mock_fix_res, mock_stream_res])
    mocker.patch("backend.agents.agentic_agent.run_python_in_sandbox", side_effect=[
        {"exit_code": 1, "stdout": "", "stderr": "NameError: name 'bad_code' is not defined", "artifacts": []},
        {"exit_code": 0, "stdout": "Success", "stderr": "", "artifacts": []}
    ])
    mock_web = mocker.patch("backend.agents.agentic_agent.search_web", return_value="Fix documentation from web")

    agent = AgenticRAGAgent()
    events = list(agent.chat_stream("Execute code block for density profile"))

    parsed = [json.loads(e.replace("data: ", "").strip()) for e in events if e.strip()]
    steps = [p.get("step") for p in parsed if p["type"] == "agent_step"]

    assert "sandbox_retry" in steps
    assert "web_complete" in steps
    assert "sandbox_complete" in steps
    assert mock_web.called
    content_deltas = [p.get("delta", "") for p in parsed if p.get("type") == "content"]
    assert any("Self-Corrected Script" in d for d in content_deltas)

def test_chat_stream_retry_and_fix_unfenced_code(mocker):
    mocker.patch("backend.agents.agentic_agent.HybridRetriever")
    mocker.patch("backend.agents.agentic_agent.insert_session", return_value=44)
    mocker.patch("backend.agents.agentic_agent.insert_message")

    # Step 2 generates bad code
    mock_codegen_res = MagicMock()
    mock_codegen_res.iter_lines.return_value = [
        json.dumps({"message": {"content": "```python\nbad_code()\n```"}}).encode("utf-8")
    ]
    # Step 2 fix returns raw unfenced code
    mock_fix_res = MagicMock()
    mock_fix_res.json.return_value = {
        "message": {"content": "import numpy as np\nfixed_code()\n"}
    }
    mock_stream_res = MagicMock()
    mock_stream_res.iter_lines.return_value = [
        json.dumps({"message": {"content": "Fixed code completed."}}).encode("utf-8")
    ]

    mocker.patch.object(AgenticRAGAgent, "_call_llm", side_effect=[mock_codegen_res, mock_fix_res, mock_stream_res])
    mock_sandbox = mocker.patch("backend.agents.agentic_agent.run_python_in_sandbox", side_effect=[
        {"exit_code": 1, "stdout": "", "stderr": "NameError: name 'bad_code' is not defined", "artifacts": []},
        {"exit_code": 0, "stdout": "Success", "stderr": "", "artifacts": []}
    ])
    mocker.patch("backend.agents.agentic_agent.search_web", return_value="Fix info")

    agent = AgenticRAGAgent()
    events = list(agent.chat_stream("Execute code block for density profile"))

    parsed = [json.loads(e.replace("data: ", "").strip()) for e in events if e.strip()]
    content_deltas = [p.get("delta", "") for p in parsed if p.get("type") == "content"]
    assert any("Self-Corrected Script" in d for d in content_deltas)
    # Ensure second sandbox call received the fixed candidate
    assert mock_sandbox.call_count == 2
    second_call_code = mock_sandbox.call_args_list[1][0][0]
    assert "fixed_code()" in second_call_code

def test_chat_stream_rag_exception(mocker):
    mock_retriever = mocker.patch("backend.agents.agentic_agent.HybridRetriever")
    mock_retriever.return_value.retrieve.side_effect = Exception("Chroma index error")
    mocker.patch("backend.agents.agentic_agent.insert_session", return_value=45)
    mocker.patch("backend.agents.agentic_agent.insert_message")

    mock_stream_res = MagicMock()
    mock_stream_res.iter_lines.return_value = [
        json.dumps({"message": {"content": "Answer without RAG."}}).encode("utf-8")
    ]
    mocker.patch.object(AgenticRAGAgent, "_call_llm", return_value=mock_stream_res)

    agent = AgenticRAGAgent()
    events = list(agent.chat_stream("Query with RAG failure"))

    parsed = [json.loads(e.replace("data: ", "").strip()) for e in events if e.strip()]
    rag_step = next(p for p in parsed if p.get("step") == "rag_error")
    assert rag_step["status"] == "warning"

def test_chat_stream_sandbox_exception(mocker):
    mock_retriever = mocker.patch("backend.agents.agentic_agent.HybridRetriever")
    mock_retriever.return_value.retrieve.return_value = "Context"
    mocker.patch("backend.agents.agentic_agent.insert_session", return_value=46)
    mocker.patch("backend.agents.agentic_agent.insert_message")

    mock_codegen_res = MagicMock()
    mock_codegen_res.json.return_value = {"message": {"content": "code"}}
    mock_stream_res = MagicMock()
    mock_stream_res.iter_lines.return_value = [
        json.dumps({"message": {"content": "Done."}}).encode("utf-8")
    ]

    mocker.patch.object(AgenticRAGAgent, "_call_llm", side_effect=[mock_codegen_res, mock_stream_res])
    mocker.patch("backend.agents.agentic_agent.run_python_in_sandbox", side_effect=Exception("Docker socket missing"))

    agent = AgenticRAGAgent()
    events = list(agent.chat_stream("Run python script"))

    parsed = [json.loads(e.replace("data: ", "").strip()) for e in events if e.strip()]
    sandbox_err = next(p for p in parsed if p.get("step") == "sandbox_error")
    assert "Docker socket missing" in sandbox_err["details"]

def test_chat_stream_llm_stream_error(mocker):
    mock_retriever = mocker.patch("backend.agents.agentic_agent.HybridRetriever")
    mock_retriever.return_value.retrieve.return_value = "Context"
    mocker.patch("backend.agents.agentic_agent.insert_session", return_value=47)
    mocker.patch("backend.agents.agentic_agent.insert_message")

    mocker.patch.object(AgenticRAGAgent, "_call_llm", side_effect=Exception("Ollama network failure"))

    agent = AgenticRAGAgent()
    events = list(agent.chat_stream("Query triggers stream error"))

    parsed = [json.loads(e.replace("data: ", "").strip()) for e in events if e.strip()]
    err_event = next(p for p in parsed if p["type"] == "error")
    assert "Ollama network failure" in err_event["error"]

def test_chat_sync_success(mocker):
    mocker.patch("backend.agents.agentic_agent.HybridRetriever")
    agent = AgenticRAGAgent()
    
    mock_events = [
        'data: {"type": "session", "session_id": 100}\n\n',
        'data: {"type": "thinking", "delta": "Thinking hard"}\n\n',
        'data: {"type": "content", "delta": "Here is the result."}\n\n',
        'data: {"type": "done", "latency_seconds": 0.8}\n\n'
    ]
    mocker.patch.object(agent, "chat_stream", return_value=iter(mock_events))

    result = agent.chat("Test sync query")
    assert result["status"] == "success"
    assert result["agent_used"] == "AgenticRAG"
    assert result["mode"] == "agentic"
    assert "Here is the result." in result["response"]
    assert "<details" in result["response"]

def test_call_llm_direct(mocker):
    mocker.patch("backend.agents.agentic_agent.HybridRetriever")
    mock_post = mocker.patch("requests.post")
    agent = AgenticRAGAgent()
    agent._call_llm("Calculate RDF", system_prompt="You are an expert", stream=True, timeout=90)
    assert mock_post.called
    args, kwargs = mock_post.call_args
    assert kwargs["stream"] is True
    assert kwargs["timeout"] == 90
    assert kwargs["json"]["messages"][0]["role"] == "system"
    assert kwargs["json"]["messages"][1]["role"] == "user"

def test_call_llm_direct_no_system(mocker):
    mocker.patch("backend.agents.agentic_agent.HybridRetriever")
    mock_post = mocker.patch("requests.post")
    agent = AgenticRAGAgent()
    agent._call_llm("Calculate RDF", stream=False)
    assert mock_post.called
    args, kwargs = mock_post.call_args
    assert kwargs["stream"] is False
    assert len(kwargs["json"]["messages"]) == 1
    assert kwargs["json"]["messages"][0]["role"] == "user"

def test_chat_stream_corrupt_chunks_and_fallbacks(mocker):
    mocker.patch("backend.agents.agentic_agent.HybridRetriever")
    mocker.patch("backend.agents.agentic_agent.insert_session", return_value=51)
    mocker.patch("backend.agents.agentic_agent.insert_message")

    # 1. Stream with empty lines, invalid JSON, thinking-only
    mock_stream_res = MagicMock()
    mock_stream_res.iter_lines.return_value = [
        b"",
        b"not json",
        json.dumps({"message": {"thinking": "Just thinking here"}}).encode("utf-8")
    ]
    mocker.patch.object(AgenticRAGAgent, "_call_llm", return_value=mock_stream_res)

    agent = AgenticRAGAgent()
    events = list(agent.chat_stream("Explain something"))
    parsed = [json.loads(e.replace("data: ", "").strip()) for e in events if e.strip()]
    done_event = next(p for p in parsed if p["type"] == "done")
    assert "Just thinking here" in done_event["full_response"]

def test_chat_stream_empty_completion(mocker):
    mocker.patch("backend.agents.agentic_agent.HybridRetriever")
    mocker.patch("backend.agents.agentic_agent.insert_session", return_value=52)
    mocker.patch("backend.agents.agentic_agent.insert_message")

    mock_stream_res = MagicMock()
    mock_stream_res.iter_lines.return_value = []
    mocker.patch.object(AgenticRAGAgent, "_call_llm", return_value=mock_stream_res)

    agent = AgenticRAGAgent()
    events = list(agent.chat_stream("Empty response"))
    parsed = [json.loads(e.replace("data: ", "").strip()) for e in events if e.strip()]
    done_event = next(p for p in parsed if p["type"] == "done")
    assert done_event["full_response"] == "Analysis completed."

def test_chat_sync_fallbacks(mocker):
    mocker.patch("backend.agents.agentic_agent.HybridRetriever")
    agent = AgenticRAGAgent()
    
    mock_events = [
        'data: corrupt-json\n\n',
        'data: {"type": "content", "delta": "Fallback content"}\n\n',
        'data: {"type": "done"}\n\n'
    ]
    mocker.patch.object(agent, "chat_stream", return_value=iter(mock_events))

    result = agent.chat("Test sync fallback")
    assert result["status"] == "success"
    assert result["response"] == "Fallback content"

def test_chat_stream_step_2_streaming(mocker):
    mocker.patch("backend.agents.agentic_agent.HybridRetriever")
    mocker.patch("backend.agents.agentic_agent.insert_session", return_value=53)
    mocker.patch("backend.agents.agentic_agent.insert_message")

    # Mock Step 2 streaming code generation with thinking and code chunks
    mock_step2_res = MagicMock()
    mock_step2_res.iter_lines.return_value = [
        json.dumps({"message": {"thinking": "Synthesizing Python RDF script..."}}).encode("utf-8"),
        json.dumps({"message": {"content": "```python\nimport ase\n"}}).encode("utf-8"),
        json.dumps({"message": {"content": "print('RDF calculated')\n```"}}).encode("utf-8"),
    ]

    # Mock Step 3 streaming synthesis
    mock_step3_res = MagicMock()
    mock_step3_res.iter_lines.return_value = [
        json.dumps({"message": {"content": "Final synthesis analysis."}}).encode("utf-8"),
    ]

    mocker.patch.object(AgenticRAGAgent, "_call_llm", side_effect=[mock_step2_res, mock_step3_res])
    mocker.patch("backend.agents.agentic_agent.run_python_in_sandbox", return_value={
        "exit_code": 0,
        "stdout": "RDF calculated",
        "stderr": "",
        "artifacts": []
    })

    agent = AgenticRAGAgent()
    events = list(agent.chat_stream("Run python script to calculate RDF"))
    parsed = [json.loads(e.replace("data: ", "").strip()) for e in events if e.strip()]

    # Verify that thinking and content events from Step 2 were emitted before sandbox_start
    types = [p["type"] for p in parsed]
    assert "thinking" in types
    assert "content" in types

    # Verify thinking delta from Step 2 was yielded
    thinking_deltas = [p["delta"] for p in parsed if p.get("type") == "thinking"]
    assert any("Synthesizing Python RDF script..." in d for d in thinking_deltas)

    # Verify code chunks from Step 2 were yielded as content events
    content_deltas = [p["delta"] for p in parsed if p.get("type") == "content"]
    assert any("import ase" in c for c in content_deltas)

def test_agentic_call_llm_with_history(mocker):
    captured_payload = {}
    def mock_post(url, json=None, **kwargs):
        nonlocal captured_payload
        captured_payload = json
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "Answer with history."}}
        return mock_resp

    mocker.patch("requests.post", side_effect=mock_post)
    mocker.patch("backend.agents.agentic_agent.HybridRetriever")

    history = [
        {"role": "user", "content": "What is Pt?"},
        {"role": "assistant", "content": "Platinum is a transition metal."}
    ]

    agent = AgenticRAGAgent()
    agent._call_llm("What is its density?", system_prompt="Custom System", history=history)

    assert "messages" in captured_payload
    msgs = captured_payload["messages"]
    assert len(msgs) == 4
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "What is Pt?"
    assert msgs[2]["content"] == "Platinum is a transition metal."
    assert msgs[3]["content"] == "What is its density?"

def test_agentic_chat_stream_passes_history(mocker):
    fake_history = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"}
    ]
    mock_get_hist = mocker.patch("backend.agents.agentic_agent.get_chat_history", return_value=fake_history)
    mocker.patch("backend.agents.agentic_agent.insert_message")
    mocker.patch("backend.agents.agentic_agent.HybridRetriever")

    called_kwargs = []
    mock_resp = MagicMock()
    mock_resp.iter_lines.return_value = [
        json.dumps({"message": {"content": "Follow-up response"}}).encode("utf-8")
    ]
    def fake_call_llm(*args, **kwargs):
        called_kwargs.append(kwargs)
        return mock_resp

    mocker.patch.object(AgenticRAGAgent, "_call_llm", side_effect=fake_call_llm)

    agent = AgenticRAGAgent()
    events = list(agent.chat_stream("Explain more about it", session_id=99))

    assert mock_get_hist.called
    assert mock_get_hist.call_args[1]["session_id"] == 99
    assert len(called_kwargs) > 0
    # Verified that history was passed to _call_llm
    assert called_kwargs[-1].get("history") == fake_history



