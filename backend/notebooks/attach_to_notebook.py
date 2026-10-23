import os
import sys
import json
import argparse
from pathlib import Path
import re
from typing import Dict, Any, Optional, List, Set

# Ensure ChatBot root is on sys.path
chatbot_root = Path(__file__).resolve().parent.parent.parent
if str(chatbot_root) not in sys.path:
    sys.path.insert(0, str(chatbot_root))

DEFAULT_NOTEBOOK_PATH = Path(__file__).parent / "running.ipynb"

def extract_notebook_context(
    notebook_path: Optional[str] = None,
    include_raw_cells: bool = False
) -> Dict[str, Any]:
    """
    Extracts the execution context of running.ipynb (or a specified notebook).
    Parses preceding setup cells, installed libraries, execution environment,
    pre-loaded global variables (e.g. `traj`, `traj_file`, `system_name`),
    and dataset files so that the model can generate fully compatible analysis scripts.
    """
    target_nb = Path(notebook_path) if notebook_path else DEFAULT_NOTEBOOK_PATH
    if not target_nb.exists():
        raise FileNotFoundError(f"Target notebook not found at: {target_nb}")

    with open(target_nb, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])

    # Locate where the generated script cell is or will be placed
    target_cell_idx = len(cells)
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", [])).strip()
            if (
                "Radial distribution function" in src or
                "Generated Scientific Analysis Script" in src or
                cell.get("metadata", {}).get("attached_by") == "SciCompute_AI"
            ):
                target_cell_idx = idx
                break

    # Analyze cells preceding target_cell_idx
    preceding_cells = cells[:target_cell_idx]

    installed_packages = []
    imports = []
    defined_variables = {}
    bash_commands = []
    file_paths = []
    system_name = None
    traj_file = None
    has_drive_mount = False
    has_traj_read = False

    for idx, cell in enumerate(preceding_cells):
        source_lines = cell.get("source", [])
        source_text = "".join(source_lines) if isinstance(source_lines, list) else str(source_lines)
        if not source_text.strip():
            continue

        cell_type = cell.get("cell_type", "code")

        if cell_type == "code":
            # Detect pip installations (%pip install or !pip install)
            pip_matches = re.findall(r"(?:%pip|!pip)\s+install\s+([a-zA-Z0-9_\- ]+)", source_text)
            for pm in pip_matches:
                for pkg in pm.split():
                    pkg_clean = pkg.strip()
                    if pkg_clean and not pkg_clean.startswith("-"):
                        installed_packages.append(pkg_clean)

            # Detect Colab drive mount
            if "google.colab" in source_text and "drive.mount" in source_text:
                has_drive_mount = True

            # Detect bash blocks
            if source_text.strip().startswith("%%bash"):
                bash_commands.append(source_text.strip())
                for p in re.findall(r'["\'](/content/[^"\']+)["\']', source_text):
                    file_paths.append(p)

            # Detect system_name
            m_sys = re.search(r'system_name\s*=\s*["\']([^"\']+)["\']', source_text)
            if m_sys:
                system_name = m_sys.group(1)
                defined_variables["system_name"] = system_name

            # Detect traj_file
            m_traj_f = re.search(r'traj_file\s*=\s*([^\n]+)', source_text)
            if m_traj_f:
                traj_file = m_traj_f.group(1).strip()
                defined_variables["traj_file"] = traj_file

            # Detect traj reading
            if re.search(r'traj\s*=\s*read\s*\(', source_text):
                has_traj_read = True
                defined_variables["traj"] = "read(traj_file, index=':')"

            # Detect imports
            for imp_line in source_text.splitlines():
                imp_line_clean = imp_line.strip()
                if imp_line_clean.startswith(("import ", "from ")):
                    imports.append(imp_line_clean)

            # Detect any other file paths in python code
            for p in re.findall(r'["\'](/content/[^"\']+)["\']', source_text):
                if p not in file_paths:
                    file_paths.append(p)

    # Resolve trajectory path if possible
    traj_path_resolved = None
    if traj_file and system_name:
        clean_traj = traj_file.replace("{system_name}", system_name).strip("'\"")
        clean_traj = re.sub(r"^[fF]['\"]", "", clean_traj).rstrip("'\"")
        traj_path_resolved = clean_traj
    elif traj_file:
        clean_traj = re.sub(r"^[fF]['\"]", "", traj_file.strip("'\"")).rstrip("'\"")
        traj_path_resolved = clean_traj

    # Detect target material system
    all_text = " ".join([
        "".join(c.get("source", [])) if isinstance(c.get("source", []), list) else str(c.get("source", ""))
        for c in preceding_cells
    ])
    detected_system = "Interfacial Water / Metal Oxide Interface"
    if "anatase" in all_text.lower() or any("anatase" in str(f).lower() for f in file_paths) or (traj_path_resolved and "anatase" in traj_path_resolved.lower()):
        detected_system = "Anatase (TiO2) / Water Interface"
    elif "iro2" in all_text.lower():
        detected_system = "IrO2 / Water Interface"

    defined_symbols = set(defined_variables.keys())
    defined_symbols.update({"read", "ase", "system_name", "traj", "traj_file"})
    if has_drive_mount:
        defined_symbols.update({"drive"})

    context = {
        "notebook_name": target_nb.name,
        "notebook_path": str(target_nb.resolve()),
        "total_cells": len(cells),
        "target_cell_index": target_cell_idx,
        "environment": {
            "platform": "Google Colab / Kaggle Cloud GPU (Linux)",
            "drive_mounted": has_drive_mount,
            "installed_packages": list(dict.fromkeys(installed_packages)),
            "referenced_paths": list(dict.fromkeys(file_paths)),
        },
        "imports": list(dict.fromkeys(imports)),
        "defined_variables": defined_variables,
        "defined_symbols": sorted(list(defined_symbols)),
        "has_preloaded_traj": has_traj_read,
        "system_name": system_name,
        "traj_file": traj_file,
        "traj_path_resolved": traj_path_resolved,
        "detected_system": detected_system,
        "preceding_cell_count": len(preceding_cells)
    }

    if include_raw_cells:
        context["preceding_cells"] = [
            {"index": idx, "cell_type": c.get("cell_type"), "source": "".join(c.get("source", []))}
            for idx, c in enumerate(preceding_cells)
        ]

    return context

def format_notebook_context_prompt(context: Dict[str, Any]) -> str:
    """
    Formats the notebook context into an authoritative prompt block for Gemma.
    Ensures Gemma knows the execution environment, pre-loaded variables,
    and system properties before generating the scientific Python script.
    """
    nb_name = context.get("notebook_name", "running.ipynb")
    cell_idx = context.get("target_cell_index", 6)
    env = context.get("environment", {})
    pkgs = ", ".join(env.get("installed_packages", ["ase", "mace-torch"])) or "ase, mace-torch"
    system_name = context.get("system_name", "molecular")
    traj_path = context.get("traj_path_resolved") or context.get("traj_file") or f"/content/MPtrj/{system_name}/Anatase__{system_name}_wrapped.extxyz"
    system_label = context.get("detected_system", "Anatase (TiO2) / Water Interface")

    return (
        f"### EXECUTION CONTEXT: JUPYTER NOTEBOOK `{nb_name}`\n"
        f"CRITICAL: You are generating a Python script that will be automatically inserted into Cell #{cell_idx + 1} of `{nb_name}` and run in Google Colab / Kaggle Cloud GPU.\n\n"
        f"1. Runtime Environment & Installed Libraries:\n"
        f"- Platform: Google Colab Linux container (`/content/`)\n"
        f"- Pre-installed Packages: `{pkgs}`, NumPy, SciPy, Matplotlib\n"
        f"- Drive / Workspace: Drive mounted at `/content/drive`, datasets extracted in `/content/MPtrj/`\n\n"
        f"2. Pre-executed Notebook State & Global Variables:\n"
        f"The preceding cells in `{nb_name}` have ALREADY executed the following code:\n"
        f"```python\n"
        f"from google.colab import drive\n"
        f"drive.mount('/content/drive')\n"
        f"# Trajectory archive extracted to /content/MPtrj/\n"
        f"%pip install ase mace-torch\n\n"
        f"from ase.io import read\n"
        f'system_name = "{system_name}"\n'
        f'traj_file = "{traj_path}"\n'
        f"traj = read(traj_file, index=':')  # Pre-loaded in notebook memory!\n"
        f"```\n\n"
        f"3. Target Physical System & Structure:\n"
        f"- Target System: {system_label}\n"
        f"- Trajectory File: `{traj_path}`\n"
        f"- Material stoichiometry: MO2 (s=2, metal=Titanium, oxide=Oxygen).\n"
        f"- Surface metal sites: Topmost Ti layer contains 5-fold Coordinatively Unsaturated Sites (Ti_cus).\n"
        f"- Surface oxygen sites: 2-fold Bridge oxygens (O_br), in-plane surface oxygens, and water oxygens (O_w).\n"
        f"- Cutoffs: CUS coordination distance ~2.5 Å; surface O-H cutoff ~1.2 Å for intact H2O vs. dissociated OH.\n\n"
        f"4. Integration Guidelines for Generated Code:\n"
        f"- Variable `traj` is ALREADY loaded in memory. You can use it directly or use safe fallback:\n"
        f"  `if 'traj' not in locals() and 'traj' not in globals(): traj = read(traj_file, index=':')`\n"
        f"- Ground your analysis on `{system_label}`.\n"
        f"- Calculate: (1) z-density profile (g/cm³), (2) pair RDFs (M_cus-O_w, O_br-H_w, O_br-O_w), "
        f"(3) survival probability P(t) over log-spaced lags via continuous set intersection, (4) surface dissociation/molecular fractions.\n"
        f"- Generate presentation figure `full_analysis_presentation.png` (2x2 layout) and `summary_report.txt`.\n"
        f"- Output ONLY valid, executable Python code in a ```python ... ``` block. No conversational preamble or fluff."
    )

def is_anatase_water_molecular_prompt(prompt: str) -> bool:
    """
    Checks if the user prompt mentions anatase ti02 water molecular
    (case-insensitive, matching 'anatase', 'tio2' or 'ti02', 'water', and 'molecular',
    or referencing Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz).
    """
    if not prompt:
        return False
    p = prompt.lower()
    if "anatase_tio2_water_molecularadsorbedwater" in p:
        return True
    has_anatase = "anatase" in p
    has_tio2 = "tio2" in p or "ti02" in p or "ti-o2" in p or "ti-02" in p or "titanium dioxide" in p
    has_water = "water" in p or "h2o" in p or "extxyz" in p
    has_molecular = "molecular" in p or "adsorbed" in p
    return has_anatase and has_tio2 and has_water and has_molecular

def is_anatase_tio2_prompt(prompt: str) -> bool:
    """
    Checks if the prompt mentions anatase tio2/ti02 (without requiring water molecular).
    """
    if not prompt:
        return False
    p = prompt.lower()
    has_anatase = "anatase" in p
    has_tio2 = "tio2" in p or "ti02" in p or "ti-o2" in p or "ti-02" in p or "titanium dioxide" in p
    return has_anatase and has_tio2

def format_anatase_extxyz_context_prompt() -> str:
    """
    Formats the context prompt for Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz
    without hardcoding naive z_min/z_max masks over all atoms.
    """
    return (
        "### AVAILABLE STRUCTURE FILE: `Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz`\n"
        "- Docker Sandbox Mount Path: `/mnt/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz`\n"
        "- Host/Workspace Fallback Path: `public/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz`\n\n"
        "CANONICAL ASE & PYTHON RULES:\n"
        "1. Loading Frame 0 vs Trajectory:\n"
        "   `atoms = read(file_path, index=0); atoms.set_pbc(True)`\n"
        "   `positions = atoms.get_positions(); symbols = np.array(atoms.get_chemical_symbols())`\n"
        "2. Distinguishing TiO2 Crystal Lattice from Adsorbed Water:\n"
        "   - Use `from ase.neighborlist import neighbor_list` with `{('O', 'H'): 1.25}` to identify water oxygens (`O_w` bonded to `H`) vs TiO2 crystal lattice oxygens (`O_slab` not bonded to `H`, coordinated to `Ti`).\n"
        "3. Identifying First Enclosing Crystal Boundary Atoms vs Slab Outer Planes:\n"
        "   - To find the first crystal boundary atoms enclosing the water molecules, find all crystal atoms (`Ti` and `O_slab`) within the first interfacial cutoff shell (`<= 3.2` Å via `neighbor_list('ijd', atoms, cutoff=3.2)`) of any water atom (`O_w` or `H`), or take the topmost/bottommost TiO2 crystal surface layers (`z_crystal >= z_crystal.max() - 1.2` or `z_crystal <= z_crystal.min() + 1.2`) excluding water atoms.\n"
        "4. Headless Plotting (Avoid 2D/3D Axes Crashes):\n"
        "   - On a standard 2D `fig, ax = plt.subplots()`, pass ONLY 2 coordinate arrays `ax.scatter(x, z, s=...)` and use `ax.set_xlabel()` / `ax.set_ylabel()`. If plotting 3D `(x, y, z)`, you MUST create a 3D axis via `ax = fig.add_subplot(111, projection='3d')`.\n"
        "   - Always save with `plt.savefig(..., dpi=300, bbox_inches='tight')` and `plt.close()`.\n"
        "5. Output Contract: Output ONLY the executable ```python ... ``` code block. NEVER fabricate or type simulated console output after the code block.\n"
        "6. Plain Text Strings (NO LaTeX in Code): NEVER use LaTeX commands or math delimiters (e.g. `\\text{}`, `\\AA`, `$ ... $`) in plot labels, titles, or code comments. Use clean Unicode/plain text like `'TiO2'`, `'H2O'`, `'X Position (Å)'`, `'Z Position (Å)'`.\n"
    )


def resolve_prompt_context(raw_text: str, notebook_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluates user prompt text against domain rules:
    - If prompt mentions 'anatase ti02 water molecular' -> provides ONLY the extxyz file location.
      The file's internal data is not sent, and notebook data is NOT sent. Does not attach to notebook.
    - If prompt just mentions 'anatase tio2' alone (for RDF, density, etc.) -> makes the notebook available
      and attaches code to running.ipynb.
    - If prompt mentions other notebook keywords -> makes notebook available.
    """
    p = raw_text.strip()
    
    # Rule 1: Anatase TiO2 Water Molecular -> only provide file location; no file data, no notebook data
    if is_anatase_water_molecular_prompt(p):
        extxyz_prompt = format_anatase_extxyz_context_prompt()
        root_path = Path(__file__).resolve().parent.parent.parent
        data_file = root_path / "public" / "Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
        if not data_file.exists():
            data_file = root_path / "ChatBot" / "public" / "Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
        
        augmented = f"{extxyz_prompt}\n\n--- USER REQUEST ---\n{raw_text}"
        return {
            "augmented_prompt": augmented,
            "matched_type": "extxyz",
            "data_file_path": str(data_file) if data_file.exists() else None,
            "notebook_context": None,
            "attach_to_notebook": False
        }

    # Rule 2: Just Anatase TiO2 alone (for RDF, density, etc.) -> make running.ipynb notebook available
    if is_anatase_tio2_prompt(p):
        nb_p = notebook_path or str(Path(__file__).parent / "running.ipynb")
        try:
            nb_ctx = extract_notebook_context(nb_p)
            nb_prompt = format_notebook_context_prompt(nb_ctx)
            augmented = f"{nb_prompt}\n\n--- USER REQUEST ---\n{raw_text}"
            return {
                "augmented_prompt": augmented,
                "matched_type": "notebook",
                "data_file_path": None,
                "notebook_context": nb_ctx,
                "attach_to_notebook": True
            }
        except Exception:
            pass

    # Check 3: General notebook keywords
    lower = p.lower()
    if any(kw in lower for kw in ["running.ipynb", "notebook", "extxyz", "traj", "density profile", "survival probability", "simulation script"]):
        nb_p = notebook_path or str(Path(__file__).parent / "running.ipynb")
        try:
            nb_ctx = extract_notebook_context(nb_p)
            nb_prompt = format_notebook_context_prompt(nb_ctx)
            augmented = f"{nb_prompt}\n\n--- USER REQUEST ---\n{raw_text}"
            return {
                "augmented_prompt": augmented,
                "matched_type": "notebook",
                "data_file_path": None,
                "notebook_context": nb_ctx
            }
        except Exception:
            pass

    return {
        "augmented_prompt": raw_text,
        "matched_type": "none",
        "data_file_path": None,
        "notebook_context": None
    }

def clean_script_code(code: str) -> str:
    """Removes markdown code fences and cleans leading/trailing whitespace."""
    if not code:
        return ""
    cleaned = code.strip()
    if cleaned.startswith("```python"):
        cleaned = cleaned[len("```python"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    # If code contains literal escaped newlines (e.g. from CLI argument), convert them
    if "\\n" in cleaned and "\n" not in cleaned:
        cleaned = cleaned.replace("\\n", "\n").replace("\\t", "\t")

    return cleaned

def attach_script_to_notebook(
    code: str,
    notebook_path: Optional[str] = None,
    replace_placeholder: bool = True,
    header_comment: Optional[str] = "# ========================================================\n# Generated Scientific Analysis Script (Gemma-4-e2b)\n# ========================================================"
) -> Dict[str, Any]:
    """
    Attaches a Python script to the Colab notebook.
    If replace_placeholder is True, searches for placeholder cells like
    '# Radial distribution function' or previously attached script cells and updates them.
    Otherwise, appends a new code cell at the end.
    """
    target_nb = Path(notebook_path) if notebook_path else DEFAULT_NOTEBOOK_PATH
    if not target_nb.exists():
        raise FileNotFoundError(f"Target notebook not found at: {target_nb}")

    with open(target_nb, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    if "cells" not in nb_data:
        nb_data["cells"] = []

    cleaned_code = clean_script_code(code)
    if not cleaned_code:
        raise ValueError("Script code cannot be empty.")

    # Prepare cell content with lines ending in \n for standard .ipynb formatting
    lines = []
    if header_comment:
        for hl in header_comment.splitlines(keepends=True):
            lines.append(hl if hl.endswith("\n") else hl + "\n")
        lines.append("\n")

    for cl in cleaned_code.splitlines(keepends=True):
        lines.append(cl if cl.endswith("\n") else cl + "\n")

    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1].rstrip("\r\n")

    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "attached_by": "SciCompute_AI",
            "language": "python"
        },
        "outputs": [],
        "source": lines
    }

    target_cell_idx = -1
    if replace_placeholder:
        # Search for placeholder cell (e.g. '# Radial distribution function' or earlier attached script)
        for idx, cell in enumerate(nb_data["cells"]):
            if cell.get("cell_type") == "code":
                src = "".join(cell.get("source", [])).strip()
                if (
                    "Radial distribution function" in src or
                    "Generated Scientific Analysis Script" in src or
                    cell.get("metadata", {}).get("attached_by") == "SciCompute_AI"
                ):
                    target_cell_idx = idx
                    break

    if target_cell_idx != -1:
        # Preserve cell ID if present
        if "id" in nb_data["cells"][target_cell_idx]:
            new_cell["id"] = nb_data["cells"][target_cell_idx]["id"]
        nb_data["cells"][target_cell_idx] = new_cell
        action_taken = "updated_cell"
    else:
        nb_data["cells"].append(new_cell)
        target_cell_idx = len(nb_data["cells"]) - 1
        action_taken = "appended_cell"

    # Clean up empty code cells
    nb_data["cells"] = [c for c in nb_data["cells"] if "".join(c.get("source", [])).strip() or c.get("cell_type") != "code"]

    with open(target_nb, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)
        f.write("\n")

    return {
        "status": "success",
        "action": action_taken,
        "notebook_path": str(target_nb.resolve()),
        "target_cell_index": target_cell_idx,
        "total_cells": len(nb_data["cells"]),
        "line_count": len(cleaned_code.splitlines())
    }

def get_latest_evaluation_code() -> Optional[str]:
    """Fetches the latest generated Python code from the SQLite evaluations table."""
    try:
        from backend.db.models import execute_query
        rows = execute_query(
            "SELECT generated_code FROM evaluations WHERE generated_code IS NOT NULL AND TRIM(generated_code) != '' ORDER BY id DESC LIMIT 1"
        )
        if rows:
            return rows[0]["generated_code"]
    except Exception as e:
        print(f"Could not fetch latest evaluation from DB: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Attach generated Python script to Colab running.ipynb")
    parser.add_argument("--code", type=str, help="Python code string to attach")
    parser.add_argument("--file", type=str, help="Path to Python script file to attach")
    parser.add_argument("--from-db", action="store_true", help="Fetch latest generated code from SQLite evaluation database")
    parser.add_argument("--notebook", type=str, default=str(DEFAULT_NOTEBOOK_PATH), help="Target .ipynb path")
    parser.add_argument("--append", action="store_true", help="Always append cell instead of replacing placeholder")
    parser.add_argument("--context", action="store_true", help="Display extracted notebook context and formatted prompt for Gemma")

    args = parser.parse_args()

    if args.context:
        ctx = extract_notebook_context(args.notebook)
        print("=== Extracted Notebook Context ===")
        print(json.dumps(ctx, indent=2))
        print("\n=== Formatted LLM Context Prompt ===")
        print(format_notebook_context_prompt(ctx))
        return

    code_to_attach = None
    if args.code:
        code_to_attach = args.code
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            code_to_attach = f.read()
    elif args.from_db:
        code_to_attach = get_latest_evaluation_code()
        if not code_to_attach:
            print("No generated code found in evaluation database.")
            return
    else:
        code_to_attach = get_latest_evaluation_code()
        if not code_to_attach:
            print("Please specify --code, --file, or --from-db.")
            return

    res = attach_script_to_notebook(
        code=code_to_attach,
        notebook_path=args.notebook,
        replace_placeholder=not args.append
    )
    print(f"Successfully attached script to {res['notebook_path']}!")
    print(f"Action: {res['action']} at cell index {res['target_cell_index']}. Total cells: {res['total_cells']}.")

if __name__ == "__main__":
    main()
