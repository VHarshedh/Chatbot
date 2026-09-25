"""
Kaggle Headless Runner for running.ipynb with GPU (T4x2)
Enables users without local GPUs to run the notebook on Kaggle cloud compute.
"""

import os
import json
import shutil
import subprocess
import argparse
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent
NOTEBOOK_FILE = NOTEBOOKS_DIR / "running.ipynb"
METADATA_FILE = NOTEBOOKS_DIR / "kernel-metadata.json"
TEMPLATE_FILE = NOTEBOOKS_DIR / "kernel-metadata.json.template"

def check_kaggle_cli() -> bool:
    """Verifies that the kaggle executable is available in PATH."""
    return shutil.which("kaggle") is not None

def check_kaggle_credentials() -> bool:
    """Verifies that ~/.kaggle/kaggle.json exists."""
    home = Path.home()
    kaggle_json = home / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()

def get_kaggle_username() -> str:
    """Attempts to read the Kaggle username from ~/.kaggle/kaggle.json."""
    home = Path.home()
    kaggle_json = home / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            with open(kaggle_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("username", "")
        except Exception:
            pass
    return ""

def generate_metadata(username: str, title: str = "Interfacial Water Analysis", slug: str = "interfacial-water-analysis"):
    """Generates kernel-metadata.json configured for GPU execution."""
    metadata = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": "running.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Generated {METADATA_FILE} with kernel ID: {username}/{slug} (enable_gpu=true)")

def push_kernel():
    """Pushes notebook to Kaggle to trigger headless execution on cloud GPU."""
    if not METADATA_FILE.exists():
        username = get_kaggle_username()
        if not username:
            print("Error: kernel-metadata.json missing and cannot detect Kaggle username.")
            print("Please create ~/.kaggle/kaggle.json or run with --username <your_username>")
            return False
        generate_metadata(username)

    print("Pushing notebook to Kaggle for GPU execution...")
    cmd = ["kaggle", "kernels", "push", "-p", str(NOTEBOOKS_DIR)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0

def check_status():
    """Checks the status of the remote Kaggle kernel execution."""
    if not METADATA_FILE.exists():
        print("kernel-metadata.json not found.")
        return
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    kernel_id = meta.get("id")
    if not kernel_id:
        print("Invalid kernel-metadata.json (no id found).")
        return
    print(f"Checking status for kernel: {kernel_id}...")
    subprocess.run(["kaggle", "kernels", "status", kernel_id])

def fetch_output(output_dir: str = "./kaggle_output"):
    """Downloads executed notebook outputs (plots, data, summary reports)."""
    if not METADATA_FILE.exists():
        print("kernel-metadata.json not found.")
        return
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    kernel_id = meta.get("id")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading artifacts for {kernel_id} to {out_path}...")
    subprocess.run(["kaggle", "kernels", "output", kernel_id, "-p", str(out_path)])

def main():
    parser = argparse.ArgumentParser(description="Kaggle CLI Cloud GPU Runner for running.ipynb")
    parser.add_argument("action", choices=["init", "push", "status", "output", "check"], help="Action to execute")
    parser.add_argument("--username", type=str, help="Kaggle username (if not auto-detected)")
    parser.add_argument("--title", type=str, default="Interfacial Water Analysis", help="Kernel title")
    parser.add_argument("--outdir", type=str, default="./kaggle_output", help="Directory to save downloaded outputs")

    args = parser.parse_args()

    if args.action == "check":
        has_cli = check_kaggle_cli()
        has_creds = check_kaggle_credentials()
        uname = get_kaggle_username()
        print(f"Kaggle CLI installed: {'YES' if has_cli else 'NO (run: pip install kaggle)'}")
        print(f"Kaggle API key (~/.kaggle/kaggle.json): {'FOUND' if has_creds else 'MISSING'}")
        if uname:
            print(f"Detected Kaggle Username: {uname}")
        return

    if args.action == "init":
        username = args.username or get_kaggle_username()
        if not username:
            print("Please provide --username <your_kaggle_username>")
            return
        generate_metadata(username, title=args.title)
        return

    if args.action == "push":
        push_kernel()
    elif args.action == "status":
        check_status()
    elif args.action == "output":
        fetch_output(args.outdir)

if __name__ == "__main__":
    main()
