import pytest
from backend.sandbox.executor import SandboxExecutor

def test_sandbox_executor_run_python(mocker):
    mock_docker = mocker.patch("docker.from_env")
    mock_client = mocker.MagicMock()
    mock_docker.return_value = mock_client
    
    mock_container = mocker.MagicMock()
    mock_container.wait.return_value = {"StatusCode": 0}
    mock_container.logs.return_value = b"Hello Sandbox"
    
    mock_client.containers.run.return_value = mock_container
    
    executor = SandboxExecutor()
    res = executor.run_python_script("print('Hello Sandbox')")
    
    assert res["exit_code"] == 0
    assert res["stdout"] == "Hello Sandbox"
    
    # Check that network_mode='none' was passed
    mock_client.containers.run.assert_called_once()
    _, kwargs = mock_client.containers.run.call_args
    assert kwargs.get("network_mode") == "none"
    assert kwargs.get("read_only") is True

def test_sandbox_executor_no_docker(mocker):
    mocker.patch("docker.from_env", side_effect=Exception("No docker"))
    executor = SandboxExecutor()
    res = executor.run_python_script("print('hi')")
    assert res["exit_code"] == 1
    assert "Docker is not running" in res["stderr"]

def test_sandbox_executor_run_python_exception(mocker):
    mock_docker = mocker.patch("docker.from_env")
    mock_docker.return_value.containers.run.side_effect = Exception("Docker run failed")
    executor = SandboxExecutor()
    res = executor.run_python_script("print('hi')")
    assert res["exit_code"] == -1
    assert "Docker run failed" in res["stderr"]
