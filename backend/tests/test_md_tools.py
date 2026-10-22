import pytest
from backend.tools.md_tools import validate_frozen_graphs, generate_lammps_input, create_batch_chain, generate_submit_chain_script, check_md_status

def test_validate_frozen_graphs(mocker):
    mock_exists = mocker.patch("pathlib.Path.exists", side_effect=[True, True, False, True, True])
    graphs = validate_frozen_graphs("dummy_dir", n_graphs=4)
    assert len(graphs) == 3 # 1 base dir + 2 valid graphs, wait the logic:
    # First call is g_dir.exists() -> True
    # Next calls are pb.exists() -> True, False, True, True
    # So 3 graphs should be appended.
    assert len(graphs) == 3

def test_validate_frozen_graphs_no_dir(mocker):
    mocker.patch("pathlib.Path.exists", return_value=False)
    graphs = validate_frozen_graphs("dummy_dir")
    assert len(graphs) == 0

@pytest.mark.parametrize("batch_num,temp,expected", [
    (1, 300, "velocity all create 300"),
    (2, 400, "read_restart restart.1.*"),
])
def test_generate_lammps_input(batch_num, temp, expected):
    params = {"temp": temp, "steps": 100}
    graphs = ["m1.pb", "m2.pb"]
    out = generate_lammps_input("deepmd", params, batch_num, graphs)
    assert expected in out
    assert "pair_style deepmd m1.pb m2.pb" in out
    assert f"fix 1 all nvt temp {temp} {temp} 0.1" in out

def test_create_batch_chain(mocker):
    mocker.patch("pathlib.Path.mkdir")
    m = mocker.patch("builtins.open", mocker.mock_open())
    dirs = create_batch_chain("out", 3, {}, ["m.pb"])
    assert len(dirs) == 3
    assert m.call_count == 3

def test_generate_submit_chain_script():
    dirs = ["batch_1", "batch_2"]
    script = generate_submit_chain_script(dirs)
    assert 'cd "batch_1"' in script
    assert "cd - > /dev/null" in script
    assert "sbatch --parsable lammps.sub" in script
    assert "--dependency=afterok:$prev_job" in script

def test_check_md_status(mocker):
    mocker.patch("pathlib.Path.glob", return_value=["restart.100"])
    status = check_md_status("dummy_dir")
    assert status["completed"] is True
    assert len(status["restart_files"]) == 1

def test_check_md_status_empty(mocker):
    mocker.patch("pathlib.Path.glob", return_value=[])
    status = check_md_status("dummy_dir")
    assert status["completed"] is False
    assert len(status["restart_files"]) == 0
