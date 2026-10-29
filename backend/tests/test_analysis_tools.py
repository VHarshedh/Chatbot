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

def test_empty_analyses_reject_missing_data():
    with pytest.raises(ValueError):
        compute_density_profile([], 1.0, [10, 10, 20])
    with pytest.raises(ValueError):
        compute_pair_rdf([], [], .1)
    with pytest.raises(ValueError):
        compute_survival_probability([], "O_w", 50)

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
    assert "Continuous survival probability computed" in report
    assert "Average hydroxyl fraction: 0.50" in report
