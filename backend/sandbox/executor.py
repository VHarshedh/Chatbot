import os
import shutil
import subprocess
import tempfile
from typing import Dict, Optional, List

try:
    import docker
except ImportError:
    docker = None

def _win_to_wsl_path(path: str) -> str:
    r"""Converts a Windows path (e.g. C:\Users\...) to a WSL mount path (/mnt/c/Users/...)."""
    abs_path = os.path.abspath(path)
    drive, rest = os.path.splitdrive(abs_path)
    if drive:
        drive_letter = drive.replace(":", "").lower()
        rest_forward = rest.replace("\\", "/")
        return f"/mnt/{drive_letter}{rest_forward}"
    return path.replace("\\", "/")

class SandboxExecutor:
    def __init__(self, image_name: str = "chatbox-sandbox:latest"):
        self.image_name = image_name
        self.client = None
        self.wsl_available = False

        # Attempt 1: Native docker Python client (Docker Desktop named pipe / socket)
        if docker:
            try:
                self.client = docker.from_env()
                self.client.ping()
            except Exception:
                self.client = None

        # Attempt 2: WSL2 Docker fallback if native client is unavailable
        if not self.client and shutil.which("wsl"):
            try:
                check = subprocess.run(
                    ["wsl", "-d", "Ubuntu", "docker", "images", "-q", self.image_name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if check.returncode == 0:
                    self.wsl_available = True
            except Exception:
                self.wsl_available = False

        self._ensure_image()

    def _ensure_image(self):
        if self.client:
            try:
                self.client.images.get(self.image_name)
            except Exception:
                print(f"Building Docker image {self.image_name}...")
                dockerfile_path = os.path.dirname(os.path.abspath(__file__))
                self.client.images.build(path=dockerfile_path, tag=self.image_name)
                print("Build complete.")

    def run_python_script(self, script_content: str, data_file_path: Optional[str] = None) -> Dict:
        """
        Runs a Python script in an isolated, secure Docker sandbox container.
        Enforces strict containment guarantees:
        - network_mode='none' (zero network access, completely isolated)
        - mem_limit='2g' (2 GB maximum RAM cap)
        - pids_limit=50 (prevent fork bombs)
        - read_only=True (root filesystem is mounted strictly read-only)
        - MPLCONFIGDIR and TMPDIR directed to writable /tmp/workspace mount
        - Data files mounted strictly read-only at /mnt/data/<basename>
        - Container auto-removes on completion
        """
        if not self.client and not self.wsl_available:
            return {
                "stdout": "",
                "stderr": "Docker is not running on the host (neither Docker Desktop nor WSL Docker daemon reachable).",
                "exit_code": 1,
                "artifacts": []
            }

        # Create temporary workspace directory on host
        workspace_dir = tempfile.mkdtemp(prefix="chatbox_workspace_")
        try:
            script_path = os.path.join(workspace_dir, "script.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)

            env_vars = {
                "MPLCONFIGDIR": "/tmp/workspace",
                "TMPDIR": "/tmp/workspace"
            }

            exit_code = 1
            logs = ""

            # Option A: Execute via Docker Python SDK
            if self.client:
                volumes = {
                    workspace_dir: {'bind': '/tmp/workspace', 'mode': 'rw'}
                }
                if data_file_path and os.path.exists(data_file_path):
                    data_basename = os.path.basename(data_file_path)
                    volumes[os.path.abspath(data_file_path)] = {'bind': f'/mnt/data/{data_basename}', 'mode': 'ro'}

                container = None
                try:
                    container = self.client.containers.run(
                        self.image_name,
                        command="python /tmp/workspace/script.py",
                        volumes=volumes,
                        environment=env_vars,
                        working_dir="/tmp/workspace",
                        network_mode="none",
                        mem_limit="2g",
                        pids_limit=50,
                        read_only=True,
                        detach=True,
                        remove=False
                    )
                    result = container.wait(timeout=120)
                    exit_code = result.get('StatusCode', 1)
                    logs = container.logs(stdout=True, stderr=True).decode('utf-8', errors='replace')
                except Exception as e:
                    logs = str(e)
                    exit_code = -1
                finally:
                    if container is not None:
                        try:
                            container.remove(force=True)
                        except Exception:
                            pass

            # Option B: Execute via WSL2 Docker CLI fallback
            elif self.wsl_available:
                wsl_workspace = _win_to_wsl_path(workspace_dir)
                cmd = [
                    "wsl", "-d", "Ubuntu", "docker", "run", "--rm",
                    "--network", "none",
                    "--memory", "2g",
                    "--pids-limit", "50",
                    "--read-only",
                    "-e", "MPLCONFIGDIR=/tmp/workspace",
                    "-e", "TMPDIR=/tmp/workspace",
                    "-w", "/tmp/workspace",
                    "-v", f"{wsl_workspace}:/tmp/workspace:rw"
                ]

                if data_file_path and os.path.exists(data_file_path):
                    data_basename = os.path.basename(data_file_path)
                    wsl_data = _win_to_wsl_path(data_file_path)
                    cmd.extend(["-v", f"{wsl_data}:/mnt/data/{data_basename}:ro"])

                cmd.extend([self.image_name, "python", "/tmp/workspace/script.py"])

                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    exit_code = proc.returncode
                    logs = proc.stdout if exit_code == 0 else (proc.stdout + "\n" + proc.stderr).strip()
                    if not logs and proc.stderr:
                        logs = proc.stderr
                except Exception as e:
                    logs = str(e)
                    exit_code = -1

            # Retrieve any plots/artifacts generated in workspace_dir
            public_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "public"))
            os.makedirs(public_dir, exist_ok=True)

            generated_files = []
            for root, dirs, files in os.walk(workspace_dir):
                for file in files:
                    # Exclude execution script and internal matplotlib cache files
                    if file == "script.py" or file.startswith("fontlist") or file.endswith(".cache"):
                        continue
                    src_path = os.path.join(root, file)
                    try:
                        dst_path = os.path.join(public_dir, file)
                        shutil.copy2(src_path, dst_path)
                        generated_files.append(dst_path)
                    except Exception:
                        generated_files.append(src_path)

            return {
                "stdout": logs if exit_code == 0 else "",
                "stderr": logs if exit_code != 0 else "",
                "logs": logs,
                "exit_code": exit_code,
                "artifacts": generated_files
            }
        finally:
            shutil.rmtree(workspace_dir, ignore_errors=True)

