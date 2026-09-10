import pytest
from backend.tools.sandbox_tools import run_python_in_sandbox, run_bash_in_sandbox

def test_run_python_in_sandbox(mocker):
    mock_executor = mocker.patch("backend.tools.sandbox_tools.SandboxExecutor")
    mock_executor.return_value.run_python_script.return_value = {"exit_code": 0, "stdout": "test"}
    res = run_python_in_sandbox("print('test')")
    assert res["exit_code"] == 0
    assert res["stdout"] == "test"

def test_run_bash_in_sandbox(mocker):
    mock_executor = mocker.patch("backend.tools.sandbox_tools.SandboxExecutor")
    mock_executor.return_value.run_python_script.return_value = {"exit_code": 0, "stdout": "test bash"}
    res = run_bash_in_sandbox("echo 'test bash'")
    assert res["exit_code"] == 0
    assert res["stdout"] == "test bash"
