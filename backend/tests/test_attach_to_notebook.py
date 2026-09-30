import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app
from backend.notebooks.attach_to_notebook import (
    attach_script_to_notebook,
    clean_script_code,
    get_latest_evaluation_code,
    DEFAULT_NOTEBOOK_PATH
)

@pytest.fixture
def temp_notebook(tmp_path):
    """Creates a temporary notebook matching running.ipynb structure."""
    nb_data = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["from google.colab import drive\n", "drive.mount('/content/drive')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "placeholder123",
                "metadata": {},
                "outputs": [],
                "source": ["# Radial distribution function"]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    nb_file = tmp_path / "running.ipynb"
    with open(nb_file, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)
    return nb_file

def test_clean_script_code():
    raw_code = "```python\nimport ase\nprint('hello')\n```"
    cleaned = clean_script_code(raw_code)
    assert cleaned == "import ase\nprint('hello')"

    raw_code_no_lang = "```\nimport numpy as np\n```"
    assert clean_script_code(raw_code_no_lang) == "import numpy as np"
    assert clean_script_code("") == ""

def test_attach_script_replaces_placeholder(temp_notebook):
    sample_script = "import ase\nfrom ase.io import read\nprint('Ready to run')"
    result = attach_script_to_notebook(
        code=sample_script,
        notebook_path=str(temp_notebook),
        replace_placeholder=True
    )

    assert result["status"] == "success"
    assert result["action"] == "updated_cell"
    assert result["target_cell_index"] == 1
    assert result["total_cells"] == 2

    # Verify content in notebook
    with open(temp_notebook, "r", encoding="utf-8") as f:
        data = json.load(f)

    cell_content = "".join(data["cells"][1]["source"])
    assert "Generated Scientific Analysis Script" in cell_content
    assert "from ase.io import read" in cell_content
    assert data["cells"][1]["metadata"]["attached_by"] == "SciCompute_AI"
    assert data["cells"][1]["id"] == "placeholder123"

def test_attach_script_appends_cell(temp_notebook):
    sample_script = "x = 42\nprint(x)"
    result = attach_script_to_notebook(
        code=sample_script,
        notebook_path=str(temp_notebook),
        replace_placeholder=False
    )

    assert result["status"] == "success"
    assert result["action"] == "appended_cell"
    assert result["target_cell_index"] == 2
    assert result["total_cells"] == 3

def test_attach_script_empty_raises(temp_notebook):
    with pytest.raises(ValueError):
        attach_script_to_notebook("", notebook_path=str(temp_notebook))

def test_attach_script_nonexistent():
    with pytest.raises(FileNotFoundError):
        attach_script_to_notebook("print(1)", notebook_path="nonexistent_nb_xyz.ipynb")

def test_api_attach_notebook_endpoint(monkeypatch):
    client = TestClient(app)
    sample_code = "import numpy as np\nprint('Endpoint test')"

    def mock_attach(code, notebook_path=None, replace_placeholder=True):
        return {
            "status": "success",
            "action": "updated_cell",
            "notebook_path": notebook_path or "mock_path.ipynb",
            "target_cell_index": 5,
            "total_cells": 6,
            "line_count": len(code.splitlines())
        }

    monkeypatch.setattr("backend.main.attach_script_to_notebook", mock_attach)

    response = client.post(
        "/api/notebooks/attach",
        json={"code": sample_code, "notebook_name": "running.ipynb"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["target_cell_index"] == 5
    assert data["total_cells"] == 6

def test_extract_notebook_context_running_ipynb():
    from backend.notebooks.attach_to_notebook import extract_notebook_context, format_notebook_context_prompt
    
    ctx = extract_notebook_context()
    assert ctx["notebook_name"] == "running.ipynb"
    assert ctx["target_cell_index"] == 6
    assert ctx["total_cells"] == 7
    assert ctx["has_preloaded_traj"] is True
    assert ctx["system_name"] == "molecular"
    assert "ase" in ctx["environment"]["installed_packages"]
    assert "mace-torch" in ctx["environment"]["installed_packages"]
    assert "traj" in ctx["defined_symbols"]
    assert "Anatase" in ctx["detected_system"]

    prompt_str = format_notebook_context_prompt(ctx)
    assert "EXECUTION CONTEXT: JUPYTER NOTEBOOK `running.ipynb`" in prompt_str
    assert "Anatase" in prompt_str
    assert "traj = read(traj_file, index=':')" in prompt_str

def test_extract_notebook_context_temp(temp_notebook):
    from backend.notebooks.attach_to_notebook import extract_notebook_context
    ctx = extract_notebook_context(str(temp_notebook), include_raw_cells=True)
    assert ctx["notebook_name"] == "running.ipynb"
    assert ctx["target_cell_index"] == 1
    assert ctx["environment"]["drive_mounted"] is True
    assert "preceding_cells" in ctx
    assert len(ctx["preceding_cells"]) == 1

def test_api_get_notebook_context_endpoint():
    client = TestClient(app)
    response = client.get("/api/notebooks/context")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["notebook_name"] == "running.ipynb"
    assert "context" in data
    assert "formatted_prompt" in data
    assert data["context"]["has_preloaded_traj"] is True

def test_resolve_prompt_context_anatase_extxyz():
    from backend.notebooks.attach_to_notebook import (
        is_anatase_water_molecular_prompt,
        is_anatase_tio2_prompt,
        resolve_prompt_context
    )

    # Test prompt with 'anatase ti02 water molecular' (with zero)
    prompt1 = "Can you analyze the anatase ti02 water molecular system?"
    assert is_anatase_water_molecular_prompt(prompt1) is True
    res1 = resolve_prompt_context(prompt1)
    assert res1["matched_type"] == "extxyz"
    assert res1["data_file_path"] is not None
    assert "Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz" in res1["augmented_prompt"]
    assert "/mnt/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz" in res1["augmented_prompt"]
    # File data must NOT be sent
    assert "426 atoms" not in res1["augmented_prompt"]
    assert "H164O202Ti60" not in res1["augmented_prompt"]
    assert "energy =" not in res1["augmented_prompt"]
    # Notebook context must NOT be sent and must NOT attach
    assert res1["notebook_context"] is None
    assert res1["attach_to_notebook"] is False

    # Test prompt with 'anatase tio2 water molecular' (with letter o)
    prompt2 = "Please run analysis on anatase tio2 water molecular adsorption"
    assert is_anatase_water_molecular_prompt(prompt2) is True
    res2 = resolve_prompt_context(prompt2)
    assert res2["matched_type"] == "extxyz"
    assert "Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz" in res2["augmented_prompt"]
    assert res2["notebook_context"] is None
    assert res2["attach_to_notebook"] is False

def test_resolve_prompt_context_anatase_tio2_notebook():
    from backend.notebooks.attach_to_notebook import (
        is_anatase_water_molecular_prompt,
        is_anatase_tio2_prompt,
        resolve_prompt_context
    )

    # Test prompt with just 'anatase tio2' (no water molecular)
    prompt = "Please setup an analysis for anatase tio2"
    assert is_anatase_water_molecular_prompt(prompt) is False
    assert is_anatase_tio2_prompt(prompt) is True

    res = resolve_prompt_context(prompt)
    assert res["matched_type"] == "notebook"
    assert res["data_file_path"] is None
    assert "running.ipynb" in res["augmented_prompt"]

