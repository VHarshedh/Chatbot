import os
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

# Define the file path with fallback mechanism
file_path = "/mnt/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
fallback_path = "public/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"

if not os.path.exists(file_path):
    if os.path.exists(fallback_path):
        file_path = fallback_path
    else:
        raise FileNotFoundError(f"Structure file not found at expected paths: {file_path} or {fallback_path}")

print(f"Loading structure from: {file_path}")

try:
    # 1. Load the structure from the first frame
    atoms = read(file_path, index=0)
    positions = atoms.get_positions()
    symbols = np.array(atoms.get_chemical_symbols())

    print(f"Successfully loaded structure. Total number of atoms: {len(positions)}")

    # 2. Identify TiO2 crystal lattice atoms (Ti and O)
    ti_indices = np.where(symbols == 'Ti')[0]
    o_indices = np.where(symbols == 'O')[0]
    
    all_crystal_indices = np.concatenate([ti_indices, o_indices])
    crystal_positions = positions[all_crystal_indices]
    crystal_symbols = symbols[all_crystal_indices]

    # 3. Identify water molecules (O-H bonds)
    # Use neighbor_list to find O-H pairs with a cutoff of 1.25 Angstroms
    water_pairs = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            atom_i_type = symbols[i]
            atom_j_type = symbols[j]
            
            # Check for O-H bond (Water)
            if (atom_i_type == 'O' and atom_j_type == 'H') or (atom_i_type == 'H' and atom_j_type == 'O'):
                water_pairs.append((i, j))

    if not water_pairs:
        print("No adsorbed water molecules (O-H bonds) found in the structure.")
    else:
        # Get positions of water molecules
        water_indices = np.unique(np.array([p for pair in water_pairs for p in pair]))
        water_positions = positions[water_indices]

        # 4. Find the bounding box of the crystal atoms enclosing the water molecules
        
        # Determine the minimum and maximum Z coordinates of the crystal atoms that enclose the water
        min_z = np.max(crystal_positions[:, 2])
        max_z = np.min(crystal_positions[:, 2])
        
        # Find the extent of the water molecules in the Z direction
        water_z_min = np.min(water_positions[:, 2])
        water_z_max = np.max(water_positions[:, 2])

        # The boundary layers are defined by the crystal atoms closest to the water extent.
        # We select crystal atoms whose Z coordinates fall within a small tolerance of the water extent.
        tolerance = 1.5  # Search window around the water molecules
        
        enclosing_crystal_indices = []
        
        for idx in all_crystal_indices:
            z = positions[idx, 2]
            # Check if the crystal atom is within the tolerance of the water extent
            if (z >= water_z_min - tolerance) and (z <= water_z_max + tolerance):
                enclosing_crystal_indices.append(idx)

        if enclosing_crystal_indices:
            enclosing_positions = positions[enclosing_crystal_indices]
            enclosing_symbols = symbols[enclosing_crystal_indices]

            # Determine the top and bottom boundary layers based on Z coordinates
            boundary_z_min = np.min(enclosing_positions[:, 2])
            boundary_z_max = np.max(enclosing_positions[:, 2])

            print("\n--- Boundary Layer Information ---")
            print(f"Enclosing crystal atoms found: {len(enclosing_crystal_indices)}")
            print(f"Top boundary layer Z position (Max Z): {boundary_z_max:.4f} Å")
            print(f"Bottom boundary layer Z position (Min Z): {boundary_z_min:.4f} Å")
            
            # Output the coordinates of the boundary atoms
            for i in range(len(enclosing_crystal_indices)):
                idx = enclosing_crystal_indices[i]
                print(f"Index: {idx}, Symbol: {enclosing_symbols[i]}, Position (x, y, z): ({positions[idx, 0]:.4f}, {positions[idx, 1]:.4f}, {positions[idx, 2]:.4f})")

except Exception as e:
    print(f"An error occurred during execution: {e}")