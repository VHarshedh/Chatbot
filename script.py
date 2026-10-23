from backend.tools.sandbox_tools import run_python_in_sandbox
import os

code = r'''import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.neighborlist import neighbor_list

# --- Configuration ---
file_path = "/mnt/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
fallback_path = "public/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
tol = 0.8  # Surface layer tolerance window in Å

# --- 1. Load Structure ---
print(f"Attempting to load structure from: {file_path}")
if not os.path.exists(file_path):
    if os.path.exists(fallback_path):
        file_path = fallback_path
    else:
        raise FileNotFoundError(f"Structure file not found at expected paths: {file_path} or {fallback_path}")

try:
    # Load the first frame of the trajectory file (index=0)
    traj = read(file_path, index=0)
    
    # Unpack based on context rules: traj is a list, atoms is the first element
    if isinstance(traj, list) and len(traj) > 0:
        atoms = traj[0]
    else:
        # Handle case where read might return a single Atoms object if index=0 was used directly
        atoms = traj

    positions = atoms.get_positions()
    symbols = np.array(atoms.get_chemical_symbols())
    
    print(f"Successfully loaded structure from: {file_path}")
    print(f"Total number of atoms: {len(positions)}")

except Exception as e:
    print(f"Error loading ASE structure: {e}")
    exit()

# --- 2. Identify Boundary Layer Atoms ---
z_min = np.min(positions[:, 2])
z_max = np.max(positions[:, 2])

# Identify atoms within the surface tolerance window (first and last layers)
# We select atoms close to the absolute minimum and maximum Z coordinates.
boundary_mask = (positions[:, 2] <= z_min + tol) | (positions[:, 2] >= z_max - tol)
boundary_atoms = atoms[boundary_mask]

if len(boundary_atoms) == 0:
    print("Warning: No boundary atoms identified using the tolerance window. Using all atoms.")
    boundary_atoms = atoms

# --- 3. Interfacial Analysis (Neighbor Search for Verification) ---
# Example neighbor search focusing on O/H interactions with the boundary layer
if len(boundary_atoms) > 0:
    print(f"\n--- Identified Boundary Atoms ({len(boundary_atoms)} total) ---")
    
    # Prepare for neighbor list analysis (using the full set for context)
    all_positions = positions
    all_symbols = symbols

    # Focus on identifying atoms that are close to water molecules (H or O)
    water_atoms_mask = np.isin(all_symbols, ["H", "O"])
    
    identified_boundary_coords = []
    
    for i, atom in enumerate(boundary_atoms):
        pos = atom.get_positions()
        atom_symbols = atom.get_chemical_symbols()
        
        # Check if this boundary atom is close to any water molecule
        is_interface = False
        for j in range(len(all_positions)):
            if water_atoms_mask[j]:
                distance = np.linalg.norm(pos - all_positions[j])
                # Define a reasonable interface cutoff (e.g., 2.5 Å)
                if distance < 2.5:
                    is_interface = True
                    break
        
        if is_interface:
            identified_boundary_coords.append({
                'index': i,
                'symbol': atom_symbols[0],
                'coords': pos.tolist()
            })

    # --- 4. Output Summary Table ---
    print("\n--- Interfacial Boundary Atoms (Closest to Water) ---")
    if identified_boundary_coords:
        header = "{:<5} {:<8} {:<10} {:<10} {:<10}".format("Idx", "Symbol", "X", "Y", "Z")
        print(header.format("Index", "Symbol", "X", "Y", "Z"))
        for data in identified_boundary_coords:
            print("{:<5} {:<8} {:<10.3f} {:<10.3f} {:<10.3f}".format(
                data['index'], 
                data['symbol'], 
                data['coords'][0], 
                data['coords'][1], 
                data['coords'][2]
            ))
    else:
        print("No specific boundary atoms identified as directly interfacing with water within the defined criteria.")


# --- 5. Generate Visualization ---
if len(boundary_atoms) > 0:
    plt.figure(figsize=(10, 8))
    
    # Plot all atoms for context (optional, but helpful)
    # scatter = plt.scatter(positions[:, 0], positions[:, 1], c=np.arange(len(positions)), cmap='viridis', s=10)

    # Highlight the identified boundary atoms in red
    boundary_pos = boundary_atoms.get_positions()
    plt.scatter(boundary_pos[:, 0], boundary_pos[:, 1], c='red', marker='o', s=50, label='Boundary Atoms')

    # Plot the entire system context (optional: plot all atoms)
    # plt.scatter(positions[:, 0], positions[:, 1], c='gray', marker='.', s=10, alpha=0.3, label='All Atoms')


    plt.title(f'2D Side-View Projection of Anatase $\\text{TiO}_2$ Boundary Layer Interfacing with Water')
    plt.xlabel('X Coordinate (Å)')
    plt.ylabel('Y Coordinate (Å)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('boundary_enclosing_atoms.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nVisualization saved to boundary_enclosing_atoms.png")
else:
    print("\nSkipping visualization as no boundary atoms were successfully identified.")
    '''

data_mount = os.path.abspath(r"public/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz")
result = run_python_in_sandbox(code, data_mount_path=data_mount)
print("=== DOCKER SANDBOX OUTPUT ===")
print("Exit Code:", result.get("exit_code"))
print("Logs:\n", result.get("logs"))
