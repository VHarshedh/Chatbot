import docker
import os
import tempfile
from typing import Dict, Optional

class SandboxExecutor:
    def __init__(self, image_name: str = "chatbox-sandbox:latest"):
        try:
            self.client = docker.from_env()
        except Exception as e:
            print(f"Warning: Docker not available. {e}")
            self.client = None
        self.image_name = image_name
        self._ensure_image()

    def _ensure_image(self):
        if not self.client: return
        try:
            self.client.images.get(self.image_name)
        except docker.errors.ImageNotFound:
            print(f"Building Docker image {self.image_name}...")
            dockerfile_path = os.path.dirname(os.path.abspath(__file__))
            self.client.images.build(path=dockerfile_path, tag=self.image_name)
            print("Build complete.")

    def run_python_script(self, script_content: str, data_file_path: Optional[str] = None) -> Dict:
        """Runs a python script in the constrained sandbox"""
        if not self.client:
            return {"stdout": "", "stderr": "Docker is not running on the host.", "exit_code": 1}

        # Create a temporary directory for the workspace (RW)
        workspace_dir = tempfile.mkdtemp(prefix="chatbox_workspace_")
        
        # Write the script into the workspace
        script_path = os.path.join(workspace_dir, "script.py")
        with open(script_path, "w") as f:
            f.write(script_content)

        volumes = {
            workspace_dir: {'bind': '/tmp/workspace', 'mode': 'rw'}
        }

        # If data is provided, mount it as Read-Only
        if data_file_path and os.path.exists(data_file_path):
            data_basename = os.path.basename(data_file_path)
            volumes[data_file_path] = {'bind': f'/mnt/data/{data_basename}', 'mode': 'ro'}
            # We assume script knows to look in /mnt/data/

        try:
            container = self.client.containers.run(
                self.image_name,
                command="python /tmp/workspace/script.py",
                volumes=volumes,
                network_mode="none",       # No internet
                mem_limit="2g",            # Max 2GB memory
                pids_limit=50,             # Prevent fork bombs
                read_only=True,            # Root FS is read-only
                detach=True,
                remove=False
            )
            
            result = container.wait(timeout=120)  # 2 minute timeout
            exit_code = result.get('StatusCode', 1)
            
            logs = container.logs(stdout=True, stderr=True).decode('utf-8')
            container.remove()

            # Retrieve any plots/files generated in workspace_dir
            generated_files = []
            for root, dirs, files in os.walk(workspace_dir):
                for file in files:
                    if file != "script.py":
                        generated_files.append(os.path.join(root, file))

            return {
                "stdout": logs if exit_code == 0 else "",
                "stderr": logs if exit_code != 0 else "",
                "exit_code": exit_code,
                "artifacts": generated_files
            }

        except Exception as e:
            return {"stdout": "", "stderr": str(e), "exit_code": -1, "artifacts": []}
