import json
import pytest
from pathlib import Path
from backend.rag.notebook_parser import (
    extract_notebook_cells,
    scan_workspace_notebooks,
    ingest_notebooks_to_vectorstore,
    register_notebooks_in_knowledge_graph,
    ingest_all_notebook_data,
)

def test_extract_notebook_cells(tmp_path):
    nb_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# Analysis Title\n", "Introductory description."]
            },
            {
                "cell_type": "code",
                "source": ["import ase\n", "atoms = ase.Atoms()"],
                "outputs": [{"output_type": "stream", "text": ["should be ignored"]}]
            },
            {
                "cell_type": "code",
                "source": []  # Empty cell should be skipped
            }
        ]
    }
    nb_file = tmp_path / "DFT" / "test_DFT_script.ipynb"
    nb_file.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_file, "w", encoding="utf-8") as f:
        json.dump(nb_content, f)

    docs = extract_notebook_cells(str(nb_file))
    assert len(docs) == 2
    assert docs[0]["metadata"]["cell_type"] == "markdown"
    assert docs[0]["metadata"]["domain"] == "DFT"
    assert "Analysis Title" in docs[0]["content"]
    assert docs[1]["metadata"]["cell_type"] == "code"
    assert "import ase" in docs[1]["content"]
    assert "should be ignored" not in docs[1]["content"]

def test_extract_notebook_cells_nonexistent():
    docs = extract_notebook_cells("nonexistent.ipynb")
    assert docs == []

def test_extract_notebook_cells_invalid_json(tmp_path):
    bad_file = tmp_path / "bad.ipynb"
    bad_file.write_text("not json content")
    docs = extract_notebook_cells(str(bad_file))
    assert docs == []

def test_scan_workspace_notebooks(tmp_path):
    (tmp_path / "DFT").mkdir()
    (tmp_path / "DFT" / "script.ipynb").write_text("{}")
    (tmp_path / ".ipynb_checkpoints").mkdir()
    (tmp_path / ".ipynb_checkpoints" / "script-checkpoint.ipynb").write_text("{}")
    (tmp_path / "venv").mkdir()
    (tmp_path / "venv" / "ignored.ipynb").write_text("{}")

    found = scan_workspace_notebooks(str(tmp_path))
    assert len(found) == 1
    assert "script.ipynb" in found[0]

def test_scan_workspace_nonexistent():
    found = scan_workspace_notebooks("nonexistent_dir_path_12345")
    assert found == []

def test_ingest_notebooks_to_vectorstore(mocker):
    mocker.patch("backend.rag.notebook_parser.scan_workspace_notebooks", return_value=["test.ipynb"])
    mocker.patch("backend.rag.notebook_parser.extract_notebook_cells", return_value=[
        {"content": "code", "metadata": {"cell_type": "code"}, "id": "cell_1"}
    ])
    mock_store = mocker.patch("backend.rag.notebook_parser.VectorStore").return_value
    
    count = ingest_notebooks_to_vectorstore(["test.ipynb"])
    assert count == 1
    assert mock_store.add_documents.called

def test_ingest_notebooks_to_vectorstore_empty(mocker):
    mocker.patch("backend.rag.notebook_parser.extract_notebook_cells", return_value=[])
    count = ingest_notebooks_to_vectorstore(["empty.ipynb"])
    assert count == 0

def test_register_notebooks_in_knowledge_graph(mocker):
    mock_kg = mocker.patch("backend.rag.notebook_parser.KnowledgeGraph").return_value
    mock_conn = mock_kg.conn

    notebooks = [
        "c:/test/DFT/calc.ipynb",
        "c:/test/Molecular Dynamics/md.ipynb",
        "c:/test/Training/train.ipynb",
        "c:/test/Analysis/analysis.ipynb"
    ]
    register_notebooks_in_knowledge_graph(notebooks)
    assert mock_conn.execute.called

def test_ingest_all_notebook_data(mocker):
    mocker.patch("backend.rag.notebook_parser.scan_workspace_notebooks", return_value=["test.ipynb"])
    mock_vec = mocker.patch("backend.rag.notebook_parser.ingest_notebooks_to_vectorstore")
    mock_kg = mocker.patch("backend.rag.notebook_parser.register_notebooks_in_knowledge_graph")

    ingest_all_notebook_data()
    assert mock_vec.called
    assert mock_kg.called
