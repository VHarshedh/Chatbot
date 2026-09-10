import pytest
from backend.agents.analysis_agent import AnalysisAgent
from backend.agents.qa_agent import QAAgent

def test_qa_agent(mocker):
    # Mock retriever class
    mock_retriever = mocker.patch("backend.agents.qa_agent.HybridRetriever")
    mock_retriever.return_value.retrieve.return_value = "Context info"
    
    # Mock requests.post to Ollama
    mock_resp = mocker.MagicMock()
    mock_resp.json.return_value = {"response": "Mocked LLM answer"}
    mocker.patch("requests.post", return_value=mock_resp)
    
    agent = QAAgent()
    ans = agent.answer("Test?")
    assert ans == "Mocked LLM answer"

def test_analysis_agent_success(mocker):
    mock_retriever = mocker.patch("backend.agents.analysis_agent.HybridRetriever")
    mock_retriever.return_value.retrieve.return_value = "Context info"
    
    # Mock LLM to return a print statement
    mock_resp = mocker.MagicMock()
    mock_resp.json.return_value = {"response": "print('hello')"}
    mocker.patch("requests.post", return_value=mock_resp)
    
    # Mock sandbox to return success
    mocker.patch("backend.agents.analysis_agent.run_python_in_sandbox", return_value={
        "exit_code": 0, "stdout": "hello", "stderr": "", "artifacts": []
    })
    
    agent = AnalysisAgent()
    res = agent.run_analysis("test")
    assert res["status"] == "success"
    assert res["stdout"] == "hello"

def test_analysis_agent_self_correction(mocker):
    mock_retriever = mocker.patch("backend.agents.analysis_agent.HybridRetriever")
    mock_retriever.return_value.retrieve.return_value = "Context info"
    
    # LLM returns code first, then fix
    mock_resp = mocker.MagicMock()
    mock_resp.json.side_effect = [{"response": "bad code"}, {"response": "good code"}]
    mocker.patch("requests.post", return_value=mock_resp)
    
    # Sandbox fails first, then succeeds
    mocker.patch("backend.agents.analysis_agent.run_python_in_sandbox", side_effect=[
        {"exit_code": 1, "stdout": "", "stderr": "Traceback:\n  File ...\nAttributeError", "artifacts": []},
        {"exit_code": 0, "stdout": "fixed", "stderr": "", "artifacts": []}
    ])
    
    # Mock web search for the error
    mocker.patch("backend.agents.analysis_agent.search_web", return_value="web fix")
    
    agent = AnalysisAgent()
    res = agent.run_analysis("test")
    
    assert res["status"] == "success"
    assert res["stdout"] == "fixed"
    assert res["code_used"] == "good code"

def test_qa_agent_exception(mocker):
    mock_retriever = mocker.patch("backend.agents.qa_agent.HybridRetriever")
    mocker.patch("requests.post", side_effect=Exception("Network error"))
    agent = QAAgent()
    ans = agent.answer("Test?")
    assert "Error connecting" in ans

def test_analysis_agent_exception(mocker):
    mock_retriever = mocker.patch("backend.agents.analysis_agent.HybridRetriever")
    mocker.patch("requests.post", side_effect=Exception("Network error"))
    from backend.agents.analysis_agent import AnalysisAgent
    agent = AnalysisAgent()
    res = agent.run_analysis("test")
    assert res["status"] == "failed"
    assert "Error connecting" in res["code_used"]
