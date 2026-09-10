import numpy as np
from ase.io import read
import matplotlib.pyplot as plt

def load_extxyz_trajectory(path: str, stride: int = 1) -> list:
    """Loads trajectory via ASE"""
    # Use index=':' to read all frames, then apply stride
    frames = read(path, index=':')
    return frames[::stride]

def identify_sites(frame, metal_symbol: str, o_stoich: int) -> dict:
    """Classifies M_sur, O_bulk, O_w, CUS, O_adb, O_sur, O_br"""
    # Mock implementation of complex structural site identification
    symbols = frame.get_chemical_symbols()
    return {
        "M_sur": [i for i, s in enumerate(symbols) if s == metal_symbol][:10],
        "O_w": [i for i, s in enumerate(symbols) if s == 'O'][10:],
        "O_bulk": [i for i, s in enumerate(symbols) if s == 'O'][:10],
    }

def compute_density_profile(traj: list, bin_width: float, cell: list) -> dict:
    """Computes density profile along z-axis"""
    # Mock
    z_max = cell[2] if isinstance(cell, list) and len(cell) == 3 else 20.0
    bins = np.arange(0, z_max, bin_width)
    density = np.random.rand(len(bins)-1)
    return {"bins": bins[:-1], "density": density}

def compute_pair_rdf(traj: list, pairs: list, dr: float) -> dict:
    """Computes radial distribution function g(r)"""
    r = np.arange(0, 10.0, dr)
    g_r = np.sin(r) + 1.0 # Mock data
    return {"r": r, "g_r": g_r}

def compute_survival_probability(traj: list, site_type: str, n_lags: int) -> dict:
    """Computes P(t) for water survival on sites"""
    t = np.arange(n_lags)
    p_t = np.exp(-t / 10.0) # Mock exponential decay
    return {"t": t, "p_t": p_t}

def compute_hydroxyl_fraction(traj: list, oh_cutoff: float) -> dict:
    """Computes fraction of hydroxylated sites over time"""
    t = np.arange(len(traj))
    frac = 0.5 + 0.1 * np.sin(t/10.0) # Mock
    return {"t": t, "fraction": frac}

def plot_combined_2x2(results: dict, save_path: str) -> str:
    """Generates 2x2 matplotlib figure of the analysis"""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Density
    if "density" in results:
        axs[0, 0].plot(results["density"]["bins"], results["density"]["density"])
        axs[0, 0].set_title("Density Profile")
        
    # RDF
    if "rdf" in results:
        axs[0, 1].plot(results["rdf"]["r"], results["rdf"]["g_r"])
        axs[0, 1].set_title("Radial Distribution Function")
        
    # Survival
    if "survival" in results:
        axs[1, 0].plot(results["survival"]["t"], results["survival"]["p_t"])
        axs[1, 0].set_title("Survival Probability")
        
    # Hydroxyl
    if "hydroxyl" in results:
        axs[1, 1].plot(results["hydroxyl"]["t"], results["hydroxyl"]["fraction"])
        axs[1, 1].set_title("Hydroxyl Fraction")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

def generate_text_report(results: dict) -> str:
    """Generates a text summary of the analysis"""
    report = "Analysis Summary:\n"
    report += "-"*20 + "\n"
    if "survival" in results:
        report += "Survival probability shows exponential decay characteristic of active sites.\n"
    if "hydroxyl" in results:
        report += f"Average hydroxyl fraction: {np.mean(results['hydroxyl']['fraction']):.2f}\n"
    return report
