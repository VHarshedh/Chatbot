import pytest
import pandas as pd
from backend.tools.dft_tools import generate_qe_scf_input, create_convergence_sweep, parse_convergence_results, plot_convergence

@pytest.fixture
def sample_structure():
    return {
        "chemical_symbols": ["Pt", "O", "H"],
        "elements": ["Pt", "O", "H"],
        "positions": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    }

@pytest.mark.parametrize("ecutwfc,kpoints,expected_str", [
    (50.0, [4, 4, 4], "ecutwfc = 50.0"),
    (80.0, [2, 2, 2], "ecutwfc = 80.0"),
])
def test_generate_qe_scf_input(sample_structure, ecutwfc, kpoints, expected_str):
    pseudo_map = {"Pt": "Pt.upf", "O": "O.upf", "H": "H.upf"}
    out = generate_qe_scf_input(sample_structure, ecutwfc, kpoints, pseudo_map)
    assert expected_str in out
    assert f"  {kpoints[0]} {kpoints[1]} {kpoints[2]} 0 0 0" in out
    assert "calculation = 'scf'" in out

def test_generate_qe_scf_input_nbnd(sample_structure):
    out = generate_qe_scf_input(sample_structure, 50.0, [2,2,2], {}, nbnd=100)
    assert "nbnd = 100" in out

def test_generate_qe_scf_input_cell_parameters(sample_structure):
    sample_structure["cell"] = [[3.94, 0.0, 0.0], [0.0, 3.94, 0.0], [0.0, 0.0, 3.94]]
    out = generate_qe_scf_input(sample_structure, 50.0, [2,2,2], {})
    assert "CELL_PARAMETERS (angstrom)" in out
    assert "3.94000000 0.00000000 0.00000000" in out

def test_create_convergence_sweep(mocker):
    mocker.patch("backend.tools.dft_tools.parse_cif", return_value={"chemical_symbols": ["Pt"], "elements": ["Pt"], "positions": [[0,0,0]]})
    mocker.patch("pathlib.Path.mkdir")
    m = mocker.patch("builtins.open", mocker.mock_open())
    
    dirs = create_convergence_sweep("dummy.cif", "ecutwfc", [40, 50], 4.0)
    assert len(dirs) == 2
    assert m.call_count == 2
    
    dirs2 = create_convergence_sweep("dummy.cif", "kpoints", [2, 4], 50.0)
    assert len(dirs2) == 2

def test_parse_convergence_results(mocker, tmp_path):
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.units import Ry
    folder = tmp_path/"ecut_50.0"
    folder.mkdir()
    (folder/"pw.out").write_text("fixture")
    atoms = Atoms("Pt")
    atoms.calc = SinglePointCalculator(atoms, energy=-12*Ry)
    reader = mocker.patch("backend.tools.dft_tools.read", return_value=atoms)
    df = parse_convergence_results(str(tmp_path))
    assert df["parameter"].iloc[0] == 50.0
    assert df["energy"].iloc[0] == pytest.approx(-12)
    assert reader.call_args.kwargs["format"] == "espresso-out"

def test_parse_convergence_results_empty(mocker):
    mocker.patch("pathlib.Path.exists", return_value=False)
    df = parse_convergence_results("dummy_dir")
    assert df.empty

def test_plot_convergence(tmp_path):
    df = pd.DataFrame([{"parameter": 1, "energy": -10}, {"parameter": 2, "energy": -20}])
    save_path = str(tmp_path / "plot.png")
    out = plot_convergence(df, "ecutwfc", save_path)
    assert out == save_path
    
def test_plot_convergence_empty():
    df = pd.DataFrame()
    out = plot_convergence(df, "ecutwfc", "plot.png")
    assert out == ""


def test_qe_uses_real_titanium_mass():
    from ase.data import atomic_masses, atomic_numbers
    structure = {"chemical_symbols": ["Ti"], "elements": ["Ti"], "positions": [[0,0,0]]}
    text = generate_qe_scf_input(structure, 60, [2,2,2], {})
    assert "ecutrho = 240" in text
    assert str(atomic_masses[atomic_numbers["Ti"]]) in text
