import pytest
from backend.tools.slurm_tools import submit_job, check_job_status, get_job_log, cancel_job

def test_submit_job(mocker):
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.stdout = "Submitted batch job 12345"
    mock_run.return_value.returncode = 0
    
    res = submit_job("test.sub")
    assert res == "12345"

def test_check_job_status(mocker):
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.stdout = "RUNNING"
    mock_run.return_value.returncode = 0
    
    status = check_job_status("12345")
    assert status["status"] == "RUNNING"

def test_get_job_log(mocker):
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.stdout = "Log output"
    mock_run.return_value.returncode = 0
    
    logs = get_job_log("12345")
    assert logs == "Log output"

def test_cancel_job(mocker):
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.stdout = ""
    mock_run.return_value.returncode = 0
    
    res = cancel_job("12345")
    assert res is True

def test_slurm_tools_exceptions(mocker):
    mock_run = mocker.patch("subprocess.run", side_effect=RuntimeError("ssh failed"))
    
    status = check_job_status("1234")
    assert status["status"] == "UNKNOWN"
    
    cancel_res = cancel_job("1234")
    assert cancel_res is False
    
    log = get_job_log("1234")
    assert "Log file not found" in log

def test_slurm_tools_conditions(mocker):
    mocker.patch("backend.tools.slurm_tools.settings.HPC_HOST", "remote")
    mock_run = mocker.patch("subprocess.run")
    
    # Job not found
    mock_run.return_value.stdout = ""
    mock_run.return_value.stderr = "Invalid job id"
    mock_run.return_value.returncode = 1
    
    status = check_job_status("9999")
    assert status["status"] == "UNKNOWN"
    
    cancel_res = cancel_job("9999")
    assert cancel_res is False
    
    # Log file not found (handled natively)
    log = get_job_log("9999")
    assert "Log file not found" in log
