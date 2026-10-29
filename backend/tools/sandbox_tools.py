from backend.sandbox.executor import SandboxExecutor
from typing import Dict, Optional

def run_python_in_sandbox(code: str, data_mount_path=None, user_prompt: str = "") -> Dict:
    """Validate before Docker and audit successful output before accepting it."""
    from backend.tools.validation_tools import validate_scientific_script_ast, audit_sandbox_execution_result
    from backend.tools.analysis_tools import inspect_structure_metadata
    validation = validate_scientific_script_ast(code, user_prompt)
    if not validation["valid"]:
        return {"stdout": "", "stderr": "Static validation failed: " + "; ".join(validation["issues"]), "exit_code": 2, "artifacts": [], "validation": validation}
    result = SandboxExecutor().run_python_script(code, data_mount_path)
    if result.get("exit_code") == 0:
        metadata = inspect_structure_metadata(data_mount_path) if isinstance(data_mount_path, str) else {}
        audit = audit_sandbox_execution_result(result.get("stdout", ""), result.get("stderr", ""), result.get("artifacts", []), user_prompt, metadata)
        result["audit"] = audit
        result["artifact_metadata"] = audit["artifact_metadata"]
        if not audit["valid"]:
            result["process_exit_code"] = 0
            result["exit_code"] = 3
            result["stderr"] = result.get("stderr", "") + "\nOutput audit failed: " + "; ".join(audit["issues"])
    return result


def run_bash_in_sandbox(script: str, data_mount_path: Optional[str] = None) -> Dict:
    """Executes a bash script inside the Docker Sandbox"""
    executor = SandboxExecutor()
    # To run bash, we can wrap the python executor or add a specific bash method to it
    # Here we simulate it by wrapping the bash command in a python subprocess for simplicity,
    # or ideally add run_bash_script to the SandboxExecutor.
    python_wrapper = f'''import subprocess
import sys
script = {script!r}
with open('script.sh', 'w') as f:
    f.write(script)
result = subprocess.run(['bash', 'script.sh'], capture_output=True, text=True)
sys.stdout.write(result.stdout)
sys.stderr.write(result.stderr)
sys.exit(result.returncode)
'''
    print("Executing Bash in Sandbox...")
    return executor.run_python_script(python_wrapper, data_mount_path)
