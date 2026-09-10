from backend.sandbox.executor import SandboxExecutor
from typing import Dict, Optional

def run_python_in_sandbox(code: str, data_mount_path: Optional[str] = None) -> Dict:
    """Executes python code inside the Docker Sandbox and returns stdout/stderr/artifacts"""
    executor = SandboxExecutor()
    print(f"Executing Python in Sandbox... Data mount: {data_mount_path}")
    return executor.run_python_script(code, data_mount_path)

def run_bash_in_sandbox(script: str, data_mount_path: Optional[str] = None) -> Dict:
    """Executes a bash script inside the Docker Sandbox"""
    executor = SandboxExecutor()
    # To run bash, we can wrap the python executor or add a specific bash method to it
    # Here we simulate it by wrapping the bash command in a python subprocess for simplicity,
    # or ideally add run_bash_script to the SandboxExecutor.
    python_wrapper = f'''import subprocess
import sys
script = r"""{script}"""
with open('script.sh', 'w') as f:
    f.write(script)
result = subprocess.run(['bash', 'script.sh'], capture_output=True, text=True)
sys.stdout.write(result.stdout)
sys.stderr.write(result.stderr)
sys.exit(result.returncode)
'''
    print("Executing Bash in Sandbox...")
    return executor.run_python_script(python_wrapper, data_mount_path)
