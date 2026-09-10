import subprocess
import re
from backend.config import settings

def _run_ssh_cmd(cmd: str) -> str:
    """Helper to run a command over SSH"""
    # For a real implementation, you would use paramiko or asyncssh.
    # We use a mock or simple subprocess here.
    host = settings.HPC_HOST
    user = settings.HPC_USER
    key = settings.HPC_KEY_PATH
    
    if host == "localhost":
        # Local mock execution for testing
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout
        
    ssh_cmd = f"ssh -i {key} {user}@{host} '{cmd}'"
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"SSH command failed: {result.stderr}")
    return result.stdout

def submit_job(script_path: str) -> str:
    """Submits a job using sbatch and returns the job ID"""
    # E.g., script_path could be an absolute path on the cluster
    cmd = f"sbatch {script_path}"
    output = _run_ssh_cmd(cmd)
    
    # Parse job ID from output: "Submitted batch job 12345"
    match = re.search(r"Submitted batch job (\d+)", output)
    if match:
        return match.group(1)
    
    # Fallback mock for local testing
    return "99999"

def check_job_status(job_id: str) -> dict:
    """Checks the status of a job using squeue/sacct"""
    cmd = f"squeue -j {job_id} -h -o %T"
    try:
        output = _run_ssh_cmd(cmd).strip()
        if output:
            return {"job_id": job_id, "status": output}
            
        # If not in squeue, check sacct
        cmd2 = f"sacct -j {job_id} -X -n -o State"
        output2 = _run_ssh_cmd(cmd2).strip()
        if output2:
            return {"job_id": job_id, "status": output2}
            
    except RuntimeError:
        pass
        
    return {"job_id": job_id, "status": "UNKNOWN"}

def cancel_job(job_id: str) -> bool:
    """Cancels a job using scancel"""
    cmd = f"scancel {job_id}"
    try:
        _run_ssh_cmd(cmd)
        return True
    except RuntimeError:
        return False

def get_job_log(job_id: str, tail_lines: int = 50) -> str:
    """Retrieves the tail of the slurm output log"""
    # We would need to know the log path, usually slurm-{job_id}.out
    cmd = f"tail -n {tail_lines} slurm-{job_id}.out"
    try:
        return _run_ssh_cmd(cmd)
    except RuntimeError:
        return "Log file not found or inaccessible."
