import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from backend.tools.analysis_tools import (
    load_extxyz_trajectory, identify_sites, compute_density_profile,
    compute_pair_rdf, compute_survival_probability, compute_hydroxyl_fraction,
    plot_combined_2x2, generate_text_report
)

def test_load_extxyz_trajectory(mocker):
    mocker.patch("backend.tools.analysis_tools.read", return_value=[1, 2, 3, 4])
    frames = load_extxyz_trajectory("dummy.extxyz", stride=2)
    assert len(frames) == 2
    assert frames == [1, 3]

def test_identify_sites(mocker):
    mock_frame = mocker.MagicMock()
    mock_frame.get_chemical_symbols.return_value = ["Pt", "O", "O", "O"]
    sites = identify_sites(mock_frame, "Pt", 1)
    assert len(sites["M_sur"]) == 1
    assert len(sites["O_bulk"]) == 3
    
def test_compute_density_profile():
    res = compute_density_profile([], 1.0, [10, 10, 20])
    assert "bins" in res
    assert "density" in res
    assert len(res["bins"]) == 19

def test_compute_pair_rdf():
    res = compute_pair_rdf([], [], 0.1)
    assert "r" in res
    assert "g_r" in res
    assert len(res["r"]) == 100

def test_compute_survival_probability():
    res = compute_survival_probability([], "O_w", 50)
    assert len(res["t"]) == 50
    assert len(res["p_t"]) == 50

def test_compute_hydroxyl_fraction():
    res = compute_hydroxyl_fraction([1, 2, 3], 1.2)
    assert len(res["t"]) == 3
    assert len(res["fraction"]) == 3

def test_plot_combined_2x2(mocker):
    mocker.patch("matplotlib.pyplot.savefig")
    results = {
        "density": {"bins": [1, 2], "density": [0.1, 0.2]},
        "rdf": {"r": [1, 2], "g_r": [1, 1]},
        "survival": {"t": [1], "p_t": [1]},
        "hydroxyl": {"t": [1], "fraction": [0.5]}
    }
    out = plot_combined_2x2(results, "dummy.png")
    assert out == "dummy.png"
    
def test_plot_combined_2x2_empty(mocker):
    mocker.patch("matplotlib.pyplot.savefig")
    out = plot_combined_2x2({}, "empty.png")
    assert out == "empty.png"

def test_generate_text_report():
    results = {
        "survival": {"t": [0], "p_t": [1]},
        "hydroxyl": {"t": [0], "fraction": [0.5]}
    }
    report = generate_text_report(results)
    assert "Survival probability shows exponential decay" in report
    assert "Average hydroxyl fraction: 0.50" in report
