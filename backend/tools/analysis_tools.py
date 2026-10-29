"""Deterministic structural analysis; distances are in angstroms."""
from collections import Counter
import numpy as np
from ase.io import read, iread
from ase.neighborlist import neighbor_list
import matplotlib.pyplot as plt


def load_extxyz_trajectory(path, stride=1):
    if stride < 1:
        raise ValueError("stride must be positive")
    return read(path, index=":")[::stride]


def separate_water_and_crystal(atoms, oh_cutoff=1.25):
    """O-H bonded phase heuristic, including hydroxyls; preserves input PBC.

    Protonated lattice oxygen is ambiguous without a material-specific model.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    i, j = neighbor_list("ij", atoms, {("O", "H"): oh_cutoff})
    oxygens = np.unique(i[(symbols[i] == "O") & (symbols[j] == "H")])
    water = np.union1d(oxygens, np.flatnonzero(symbols == "H"))
    crystal = np.setdiff1d(np.arange(len(atoms)), water)
    return {"water_indices": water.tolist(), "water_oxygen_indices": oxygens.tolist(), "crystal_indices": crystal.tolist(),
            "method": "O-H bonding heuristic; protonated lattice oxygen may be ambiguous"}


def find_enclosing_crystal_boundary(atoms, tolerance=0.8):
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    indices = np.array(separate_water_and_crystal(atoms)["crystal_indices"], dtype=int)
    if not len(indices):
        return []
    z = atoms.positions[indices, 2]
    return indices[(z <= z.min() + tolerance) | (z >= z.max() - tolerance)].tolist()


def inspect_structure_metadata(file_path):
    try:
        atoms = read(file_path, index=0)
        phases = separate_water_and_crystal(atoms)
        summary = {"water_atom_count": len(phases["water_indices"]), "water_oxygen_count": len(phases["water_oxygen_indices"]),
                   "crystal_slab_atom_count": len(phases["crystal_indices"]), "method": phases["method"]}
        for phase in ("water", "crystal"):
            z = atoms.positions[phases[phase + "_indices"], 2]
            summary[phase + "_z_range_A"] = [float(z.min()), float(z.max())] if len(z) else []
        return {"file_path": str(file_path), "frame_count": sum(1 for _ in iread(file_path)), "total_atoms_frame_0": len(atoms),
                "chemical_formula": atoms.get_chemical_formula(), "element_counts": dict(Counter(atoms.get_chemical_symbols())),
                "cell_lengths_angles": atoms.cell.cellpar().tolist(), "pbc": atoms.pbc.tolist(),
                "info_keys": list(atoms.info), "arrays_keys": list(atoms.arrays), "phase_partition_preview": summary}
    except Exception as exc:
        return {"file_path": str(file_path), "error": str(exc)}


def identify_sites(frame, metal_symbol, o_stoich, surface_depth=1.5, adsorption_cutoff=2.5):
    symbols = np.array(frame.get_chemical_symbols())
    metal = np.flatnonzero(symbols == metal_symbol)
    phase = separate_water_and_crystal(frame)
    bulk = [i for i in phase["crystal_indices"] if symbols[i] == "O"]
    water = phase["water_oxygen_indices"]
    surface = metal[frame.positions[metal, 2] >= frame.positions[metal, 2].max() - surface_depth] if len(metal) else metal
    cus, adsorbed = set(), set()
    for m in surface:
        for o in water:
            if frame.get_distance(int(m), o, mic=True) <= adsorption_cutoff:
                cus.add(int(m)); adsorbed.add(o)
    top = max(frame.positions[metal, 2]) if len(metal) else float("inf")
    oxygen_surface = [o for o in bulk if frame.positions[o, 2] >= top - surface_depth]
    return {"M_sur": surface.tolist(), "O_bulk": bulk, "O_w": water, "M_cus": sorted(cus),
            "O_adb": sorted(adsorbed), "O_sur": oxygen_surface,
            "O_br": [o for o in oxygen_surface if sum(frame.get_distance(o, int(m), mic=True) <= adsorption_cutoff for m in metal) >= 2],
            "stoichiometry_matches": len(bulk) == o_stoich * len(metal)}


def compute_density_profile(traj, bin_width, cell):
    if not traj or bin_width <= 0:
        raise ValueError("nonempty trajectory and positive bin_width required")
    from ase.cell import Cell
    lattice = Cell.new(cell)
    height = abs(np.linalg.det(lattice)) / np.linalg.norm(np.cross(lattice[0], lattice[1]))
    edges = np.linspace(0, height, int(np.ceil(round(height / bin_width, 12))) + 1)
    mass = np.zeros(len(edges)-1)
    for frame in traj:
        z = frame.get_scaled_positions(wrap=True)[:, 2] * height
        mass += np.histogram(z, edges, weights=frame.get_masses())[0]
    volumes = abs(np.linalg.det(lattice)) * np.diff(edges) / height
    return {"bins": (edges[:-1]+edges[1:])/2, "density": mass / len(traj) / volumes * 1.66053906660}


def compute_pair_rdf(traj, pairs, dr):
    if not traj or not pairs or dr <= 0:
        raise ValueError("trajectory, element pairs and positive dr required")
    if any(not np.all(f.pbc) or f.get_volume() <= 0 for f in traj):
        raise ValueError("bulk RDF requires fully periodic cells")
    radius = min(min(1 / np.linalg.norm(f.cell.reciprocal(), axis=1)) / 2 for f in traj)
    edges = np.arange(0, radius + dr * 0.01, dr)
    if len(edges) < 2:
        raise ValueError("dr exceeds minimum image radius")
    shell = 4*np.pi/3*np.diff(edges**3)
    values = []
    for a, b in pairs:
        hist, norm = np.zeros(len(shell)), np.zeros(len(shell))
        for frame in traj:
            syms = np.array(frame.get_chemical_symbols())
            ia, ib = np.flatnonzero(syms == a), np.flatnonzero(syms == b)
            distances = frame.get_all_distances(mic=True)[np.ix_(ia, ib)]
            distances = distances[ia[:, None] != ib[None, :]]
            hist += np.histogram(distances, edges)[0]
            norm += len(ia)*(len(ib)-(a == b))/frame.get_volume()*shell
        if np.any(norm <= 0):
            raise ValueError("pair species missing or insufficient atoms")
        values.append(hist/norm)
    return {"r": (edges[:-1]+edges[1:])/2, "g_r": values[0] if len(values)==1 else np.array(values), "pairs": pairs}


def compute_survival_probability(traj, site_type, n_lags):
    """Continuous membership survival; frames must preserve atom ordering."""
    if not traj or n_lags < 1:
        raise ValueError("trajectory and positive n_lags required")
    sets = []
    for frame in traj:
        metals = [s for s in frame.get_chemical_symbols() if s not in {"H", "O"}]
        if site_type == "O_w":
            sets.append(set(separate_water_and_crystal(frame)["water_oxygen_indices"]))
        elif metals:
            sets.append(set(identify_sites(frame, metals[0], 2)[site_type]))
        else:
            raise ValueError("site type requires a metal-containing structure")
    probabilities = []
    for lag in range(min(n_lags, len(sets))):
        samples = [len(set.intersection(*sets[t:t+lag+1]))/len(sets[t]) for t in range(len(sets)-lag) if sets[t]]
        probabilities.append(np.mean(samples) if samples else np.nan)
    return {"t": np.arange(len(probabilities)), "p_t": np.array(probabilities)}


def compute_hydroxyl_fraction(traj, oh_cutoff):
    fractions = []
    for frame in traj:
        symbols = np.array(frame.get_chemical_symbols())
        oxygens = np.flatnonzero(symbols == "O")
        i, j = neighbor_list("ij", frame, {("O", "H"): oh_cutoff})
        counts = Counter(i[(symbols[i] == "O") & (symbols[j] == "H")])
        fractions.append(sum(counts[o] == 1 for o in oxygens)/len(oxygens) if len(oxygens) else np.nan)
    return {"t": np.arange(len(traj)), "fraction": np.array(fractions)}


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
        report += "Continuous survival probability computed from persistent site membership.\n"
    if "hydroxyl" in results:
        report += f"Average hydroxyl fraction: {np.mean(results['hydroxyl']['fraction']):.2f}\n"
    return report
