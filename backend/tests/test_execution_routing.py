import json

import pytest
from fastapi.testclient import TestClient

from backend.main import app


BOUNDARY_REQUEST = (
    "I need you to create a python script that parses the anatase tio2 molecular "
    "extxyz file and returns the positions of the atom on the boundary of the "
    "lattice configuration as output after running the code in the secure docker container."
)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("message,expected", [
    (BOUNDARY_REQUEST, True),
    ("Create a python file that parses the anatase ti02 molecular adsorbed water extxyz file and returns the coordinates of the atoms that enclose the water molecules in the crystal configuration. Only the first boundary atoms enclosing the water molecules must be returned in the python file that must be executed in a secure docker container and the output shown here.", True),
    ("Please execute the generated Python script.", True),
    ("Run the analysis in Docker.", True),
    ("Write a Python script for anatase tio2 molecular extxyz.", False),
    ("Explain how to run Python in Docker.", False),
    ("What is a running Docker container?", False),
    ("Write the script but don't run it.", False),
    ("Create code without executing it.", False),
    ("Only show the code to run in Docker.", False),
])
def test_single_shot_execution_routing(mocker, streaming, message, expected):
    # Added notebook/data context may mention execution even for code-only requests.
    mocker.patch("backend.main.resolve_prompt_context", return_value={
        "matched_type": "extxyz", "augmented_prompt": "Run in Docker.\n" + message,
    })
    method = "chat_stream" if streaming else "chat"
    result = iter(['data: {"type": "done"}\n\n']) if streaming else {"status": "success"}
    chat = mocker.patch(f"backend.main.SingleShotAgent.{method}", return_value=result)
    endpoint = "/api/chat/stream" if streaming else "/api/chat"
    response = TestClient(app).post(endpoint, json={"message": message})
    assert response.status_code == 200
    assert chat.call_args.kwargs["run_sandbox"] is expected


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mode", ["agentic", "agentic_rag"])
def test_agentic_keeps_own_execution_routing(mocker, streaming, mode):
    mocker.patch("backend.agents.agentic_agent.HybridRetriever")
    method = "chat_stream" if streaming else "chat"
    result = iter(['data: {"type": "done"}\n\n']) if streaming else {"status": "success"}
    chat = mocker.patch(f"backend.main.AgenticRAGAgent.{method}", return_value=result)
    endpoint = "/api/chat/stream" if streaming else "/api/chat"
    response = TestClient(app).post(endpoint, json={"message": BOUNDARY_REQUEST, "mode": mode})
    assert response.status_code == 200
    assert "run_sandbox" not in chat.call_args.kwargs


@pytest.mark.parametrize("streaming", [False, True])
def test_boundary_request_reaches_sandbox(mocker, streaming):
    mocker.patch("backend.agents.baseline_agent.get_chat_history", return_value=[])
    mocker.patch("backend.agents.baseline_agent.insert_session", return_value=37)
    mocker.patch("backend.agents.baseline_agent.insert_message")
    mocker.patch("backend.rag.chat_memory.get_chat_memory_store")
    code = "from ase.io import read\natoms = read('/mnt/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz')\nprint(atoms.positions)"
    llm = mocker.MagicMock()
    llm.iter_lines.return_value = [json.dumps({
        "message": {"content": f"```python\n{code}\n```"}, "done": True,
    }).encode()]
    mocker.patch("backend.agents.baseline_agent.requests.post", return_value=llm)
    sandbox = mocker.patch("backend.tools.sandbox_tools.run_python_in_sandbox", return_value={
        "exit_code": 0, "stdout": "BOUNDARY_COORDINATES_FROM_SANDBOX", "stderr": "", "artifacts": [],
    })
    endpoint = "/api/chat/stream" if streaming else "/api/chat"
    response = TestClient(app).post(endpoint, json={"message": BOUNDARY_REQUEST})
    assert response.status_code == 200
    sandbox.assert_called_once()
    assert sandbox.call_args.args[0] == code
    assert sandbox.call_args.kwargs["data_mount_path"].endswith("Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz")
    assert "BOUNDARY_COORDINATES_FROM_SANDBOX" in response.text
