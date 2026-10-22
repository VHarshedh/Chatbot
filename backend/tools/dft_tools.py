import os
from pathlib import Path
from ase.io import read
import pandas as pd
import matplotlib.pyplot as plt

def parse_cif(cif_path: str) -> dict:
    """Reads CIF via ASE, returns elements, cell params, positions"""
    atoms = read(cif_path)
    return {
        "elements": list(set(atoms.get_chemical_symbols())),
        "cell": atoms.get_cell().tolist(),
        "positions": atoms.get_positions().tolist(),
        "chemical_symbols": atoms.get_chemical_symbols()
    }

def generate_qe_scf_input(structure: dict, ecutwfc: float, kpoints: list, pseudo_map: dict, nbnd: int = None, smearing: str = 'mv', degauss: float = 0.01) -> str:
    """Generates pw.in content for Quantum ESPRESSO"""
    # Simplified generator for demonstration
    symbols = structure["chemical_symbols"]
    elements = structure["elements"]
    
    input_str = f"&CONTROL\n  calculation = 'scf',\n  pseudo_dir = './pseudo/',\n  outdir = './out/'\n/\n"
    input_str += f"&SYSTEM\n  ibrav = 0,\n  nat = {len(symbols)},\n  ntyp = {len(elements)},\n  ecutwfc = {ecutwfc},\n"
    if nbnd:
        input_str += f"  nbnd = {nbnd},\n"
    input_str += f"  occupations = 'smearing',\n  smearing = '{smearing}',\n  degauss = {degauss}\n/\n"
    input_str += f"&ELECTRONS\n  conv_thr = 1.0d-8\n/\n"
    
    input_str += "ATOMIC_SPECIES\n"
    for el in elements:
        pseudo = pseudo_map.get(el, f"{el}.UPF")
        # Example atomic masses
        mass = 1.0 if el == 'H' else 16.0 if el == 'O' else 195.08 if el == 'Pt' else 1.0
        input_str += f"  {el} {mass} {pseudo}\n"
        
    if "cell" in structure and structure["cell"]:
        input_str += "CELL_PARAMETERS (angstrom)\n"
        for vec in structure["cell"]:
            input_str += f"  {vec[0]:.8f} {vec[1]:.8f} {vec[2]:.8f}\n"

    input_str += "ATOMIC_POSITIONS (angstrom)\n"
    for sym, pos in zip(symbols, structure["positions"]):
        input_str += f"  {sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n"
        
    input_str += "K_POINTS (automatic)\n"
    input_str += f"  {kpoints[0]} {kpoints[1]} {kpoints[2]} 0 0 0\n"
    
    return input_str

def create_convergence_sweep(cif_path: str, mode: str, values: list, fixed_param: float) -> list:
    """Creates job directories for convergence sweeps"""
    # mode is either 'ecutwfc' or 'kpoints'
    base_dir = Path("convergence_sweep")
    base_dir.mkdir(exist_ok=True)
    job_dirs = []
    
    structure = parse_cif(cif_path)
    pseudo_map = {"Pt": "Pt_ONCV_PBE-1.2.upf", "O": "O.pbe-n-kjpaw_psl.0.1.UPF", "H": "H.pbe-kjpaw_psl.0.1.UPF"}
    
    for val in values:
        if mode == "ecutwfc":
            ecutwfc = val
            kpts = [int(fixed_param), int(fixed_param), int(fixed_param)]
            dir_name = f"ecut_{val}"
        else:
            ecutwfc = fixed_param
            kpts = [int(val), int(val), int(val)]
            dir_name = f"kpt_{val}"
            
        job_dir = base_dir / dir_name
        job_dir.mkdir(exist_ok=True)
        job_dirs.append(job_dir)
        
        inp = generate_qe_scf_input(structure, ecutwfc, kpts, pseudo_map)
        with open(job_dir / "pw.in", "w") as f:
            f.write(inp)
            
    return job_dirs

def parse_convergence_results(sweep_dir: str) -> pd.DataFrame:
    """Reads pw.out files, returns energy table. Mock implementation."""
    # In reality, this would regex parse '!' total energy from QE out files
    data = []
    sweep_path = Path(sweep_dir)
    if sweep_path.exists():
        for job_dir in sweep_path.iterdir():
            if job_dir.is_dir():
                # Mock parsing
                val = float(job_dir.name.split("_")[1])
                # Mock energy that converges
                energy = -100.0 - (1.0 / val)
                data.append({"parameter": val, "energy": energy})
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values("parameter")
    return df

def plot_convergence(df: pd.DataFrame, mode: str, save_path: str) -> str:
    """Generates publication-quality convergence plot"""
    if df.empty:
        return ""
    plt.figure(figsize=(8, 6))
    plt.plot(df["parameter"], df["energy"], marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel(f"{mode.capitalize()} Parameter", fontsize=14)
    plt.ylabel("Total Energy (Ry)", fontsize=14)
    plt.title(f"{mode.capitalize()} Convergence", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path
