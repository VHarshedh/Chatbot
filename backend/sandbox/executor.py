import os
import shutil
import subprocess
import tempfile
import uuid
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
        self.cli_available = False

        # Attempt 1: Native docker Python client (Docker Desktop named pipe / socket)
        if docker:
            try:
                self.client = docker.from_env()
                self.client.ping()
            except Exception:
                self.client = None

        # Respect Docker Desktop's active CLI context when SDK named-pipe discovery fails.
        if not self.client and shutil.which("docker"):
            try:
                check = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True, text=True, timeout=10)
                self.cli_available = check.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                pass

        # Attempt 2: WSL2 Docker fallback if native client is unavailable
        if not self.client and not self.cli_available and shutil.which("wsl"):
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
        if not self.client and not self.wsl_available and not getattr(self, "cli_available", False):
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
                "TMPDIR": "/tmp/workspace",
                "PYTHONPATH": "/opt/toolkit"
            }

            exit_code = 1
            stdout, stderr = "", ""
            paths = [data_file_path] if isinstance(data_file_path, str) else list(data_file_path or [])
            names = [os.path.basename(os.path.abspath(p)) for p in paths]
            if len(names) != len(set(names)):
                raise ValueError("Input mount basenames must be unique")

            # Option A: Execute via Docker Python SDK
            if self.client:
                volumes = {
                    workspace_dir: {'bind': '/tmp/workspace', 'mode': 'rw'},
                    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')): {'bind': '/opt/toolkit/backend/tools', 'mode': 'ro'},
                    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.py')): {'bind': '/opt/toolkit/backend/config.py', 'mode': 'ro'}
                }
                for path in paths:
                    if not os.path.exists(path):
                        raise FileNotFoundError(path)
                    volumes[os.path.abspath(path)] = {'bind': f'/mnt/data/{os.path.basename(os.path.abspath(path))}', 'mode': 'ro'}

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
                    from backend.agents.cancellation import on_cancel, check_cancelled
                    on_cancel(lambda: container.remove(force=True))
                    check_cancelled()
                    result = container.wait(timeout=120)
                    exit_code = result.get('StatusCode', 1)
                    stdout = container.logs(stdout=True, stderr=False).decode('utf-8', errors='replace')
                    stderr = container.logs(stdout=False, stderr=True).decode('utf-8', errors='replace')
                except Exception as e:
                    stderr = str(e)
                    exit_code = -1
                finally:
                    if container is not None:
                        try:
                            container.remove(force=True)
                        except Exception:
                            pass

            # Option B: Execute via WSL2 Docker CLI fallback
            elif self.wsl_available or getattr(self, "cli_available", False):
                convert_path = _win_to_wsl_path if self.wsl_available else os.path.abspath
                wsl_workspace = convert_path(workspace_dir)
                prefix = ["wsl", "-d", "Ubuntu", "docker"] if self.wsl_available else ["docker"]
                container_name = "chatbox-" + uuid.uuid4().hex
                cmd = prefix + ["run", "--rm", "--name", container_name,
                    "--network", "none",
                    "--memory", "2g",
                    "--pids-limit", "50",
                    "--read-only",
                    "-e", "MPLCONFIGDIR=/tmp/workspace",
                    "-e", "TMPDIR=/tmp/workspace",
                    "-e", "PYTHONPATH=/opt/toolkit",
                    "-v", f"{convert_path(os.path.join(os.path.dirname(__file__), "..", "tools"))}:/opt/toolkit/backend/tools:ro",
                    "-v", f"{convert_path(os.path.join(os.path.dirname(__file__), "..", "config.py"))}:/opt/toolkit/backend/config.py:ro",
                    "-w", "/tmp/workspace",
                    "-v", f"{wsl_workspace}:/tmp/workspace:rw"
                ]

                for path in paths:
                    if not os.path.exists(path):
                        raise FileNotFoundError(path)
                    cmd.extend(["-v", f"{convert_path(path)}:/mnt/data/{os.path.basename(os.path.abspath(path))}:ro"])

                cmd.extend([self.image_name, "python", "/tmp/workspace/script.py"])

                try:
                    from backend.agents.cancellation import run_cancellable_process
                    proc = run_cancellable_process(cmd, timeout=120,
                        cleanup=lambda: subprocess.run(prefix + ["rm", "-f", container_name], capture_output=True, timeout=10))
                    exit_code = proc.returncode
                    stdout, stderr = proc.stdout, proc.stderr
                except Exception as e:
                    stderr = str(e)
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
                    if os.path.islink(src_path) or not os.path.isfile(src_path):
                        continue
                    try:
                        dst_path = os.path.join(public_dir, file)
                        shutil.copy2(src_path, dst_path)
                        generated_files.append(dst_path)
                    except OSError:
                        continue

            return {
                "stdout": stdout,
                "stderr": stderr,
                "logs": stdout + stderr,
                "exit_code": exit_code,
                "artifacts": generated_files
            }
        finally:
            shutil.rmtree(workspace_dir, ignore_errors=True)

