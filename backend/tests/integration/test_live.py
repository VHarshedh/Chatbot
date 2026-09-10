import pytest
from backend.agents.qa_agent import QAAgent
from backend.tools.sandbox_tools import run_python_in_sandbox

@pytest.mark.skip(reason="Integration tests require live Ollama and Docker")
def test_live_qa_agent():
    # This will hit the actual Ollama instance
    agent = QAAgent()
    ans = agent.answer_question("What is Quantum ESPRESSO?")
    assert len(ans) > 10

@pytest.mark.skip(reason="Integration tests require live Docker")
def test_live_sandbox():
    # This will hit the actual local Docker daemon
    res = run_python_in_sandbox("print('hello live sandbox')")
    assert res["exit_code"] == 0
    assert "hello live sandbox" in res["stdout"]
