import pytest
from backend.tools.training_tools import scan_raw_systems, validate_raw_data, create_test_split, raw_to_set, generate_deepmd_config, create_training_jobs

def test_scan_raw_systems(mocker):
    # Mock base_dir.exists() -> True
    mocker.patch("pathlib.Path.exists", return_value=True)
    # Mock iterdir to return items
    mock_item1 = mocker.MagicMock()
    mock_item1.is_dir.return_value = True
    mock_item1.name = "system1_Water"
    mock_item1.__str__.return_value = "system1_Water"
    
    mock_item2 = mocker.MagicMock()
    mock_item2.is_dir.return_value = True
    mock_item2.name = "system2_Ice"
    mock_item2.__str__.return_value = "system2_Ice"
    
    mocker.patch("pathlib.Path.iterdir", return_value=[mock_item1, mock_item2])
    
    res = scan_raw_systems("dummy", "_Water")
    assert len(res) == 1
    assert "system1_Water" in res[0]

def test_validate_raw_data(mocker):
    # Test valid
    mocker.patch("pathlib.Path.exists", return_value=True)
    res = validate_raw_data("dummy")
    assert res["valid"] is True
    assert res["has_box"] is True
    
    # Test invalid
    mocker.patch("pathlib.Path.exists", side_effect=[True, False, True, True, True])
    res = validate_raw_data("dummy2")
    assert res["valid"] is False
    assert res["has_box"] is True
    assert res["has_coord"] is False

def test_create_test_split(mocker):
    mocker.patch("pathlib.Path.mkdir")
    out = create_test_split("sys")
    assert "Test" in out

def test_raw_to_set(mocker):
    mocker.patch("pathlib.Path.mkdir")
    sets = raw_to_set("sys")
    assert len(sets) == 3
    assert "set.000" in sets[0]

@pytest.mark.parametrize("seed,lr", [
    (123, 0.005),
    (456, 0.001)
])
def test_generate_deepmd_config(seed, lr):
    config = generate_deepmd_config(["H", "O"], [10, 10], seed=seed, lr=lr)
    assert config["model"]["type_map"] == ["H", "O"]
    assert config["model"]["descriptor"]["seed"] == seed
    assert config["learning_rate"]["start_lr"] == lr

def test_create_training_jobs(mocker):
    mocker.patch("pathlib.Path.mkdir")
    m = mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("json.dump")
    
    dirs = create_training_jobs("sys", 2, [1])
    assert len(dirs) == 2
    assert m.call_count == 2
