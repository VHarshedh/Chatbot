from backend.tools.sandbox_tools import run_python_in_sandbox
import os

code = '''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import read
from ase.neighborlist import neighbor_list

# --- Configuration ---
file_path = "/mnt/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
cutoff = 3.2  # Cutoff for neighbor search (Å)
oh_cutoff = 1.25 # Cutoff for O-H bond identification (Å)

# 1. Load the structure
try:
    atoms = read(file_path, index=0)
    atoms.set_pbc(True)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Attempting fallback.")
    file_path = "public/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
    try:
        atoms = read(file_path, index=0)
        atoms.set_pbc(True)
    except FileNotFoundError:
        print("Error: Fallback file also not found. Exiting.")
        exit()

positions = atoms.get_positions()
symbols = np.array(atoms.get_chemical_symbols())

# 2. Separate Water atoms (O_w, H) from Anatase TiO2 crystal slab atoms (Ti, O_slab)
# Find O-H covalent bonds within 1.25 Å
i_oh, j_oh = neighbor_list('ij', atoms, {('O', 'H'): oh_cutoff})

# Water atoms are all H atoms plus the unique O atoms involved in O-H bonds
water_indices = set(np.where(symbols == 'H')[0])
for i in i_oh:
    water_indices.add(i)

# Crystal slab atoms are Ti and O atoms that are NOT water atoms
crystal_indices = set()
for i in range(len(symbols)):
    if symbols[i] in ('Ti', 'O'):
        # Check if this atom is NOT part of the water group
        if i not in water_indices:
            crystal_indices.add(i)

# Convert sets to sorted lists for indexing
water_indices = sorted(list(water_indices))
crystal_indices = sorted(list(crystal_indices))

print(f"Total Atoms: {len(symbols)}")
print(f"Water Atoms (O_w + H): {len(water_indices)}")
print(f"Crystal Slab Atoms (Ti + O_slab): {len(crystal_indices)}\\n")


# 3. Find ONLY the first boundary atoms of the TiO2 crystal that enclose the water molecules
nl_i, nl_j, nl_d = neighbor_list('ijd', atoms, cutoff=cutoff)
crystal_set = set(crystal_indices)
water_set = set(water_indices)

# Filter for crystal-water interactions
involved_crystal_dict = {}
for i_atom, j_atom, dist in zip(nl_i, nl_j, nl_d):
    if i_atom in crystal_set and j_atom in water_set:
        if i_atom not in involved_crystal_dict or dist < involved_crystal_dict[i_atom]:
            involved_crystal_dict[i_atom] = dist

if not involved_crystal_dict:
    print("No crystal-water interactions found within the cutoff distance.")
    boundary_enclosing_atoms = []
    first_boundary_indices = []
else:
    boundary_enclosing_atoms = []
    for i_atom, min_dist in involved_crystal_dict.items():
        boundary_enclosing_atoms.append({
            'ID': i_atom,
            'Symbol': symbols[i_atom],
            'X': positions[i_atom, 0],
            'Y': positions[i_atom, 1],
            'Z': positions[i_atom, 2],
            'Min_Dist_to_Water': min_dist
        })

    # Sort by Z coordinate to identify the first boundary layer
    boundary_enclosing_atoms.sort(key=lambda x: x['Z'])
    first_boundary_indices = [item['ID'] for item in boundary_enclosing_atoms]


# 4. Print a summary header and formatted table
print("--- First Boundary Layer Analysis ---")
print(f"Total Atoms: {len(symbols)}")
print(f"Crystal Slab Atoms: {len(crystal_indices)}")
print(f"Water Atoms: {len(water_indices)}")
print(f"Number of Crystal Atoms Enclosing Water (First Boundary): {len(first_boundary_indices)}\\n")

if boundary_enclosing_atoms:
    df = pd.DataFrame(boundary_enclosing_atoms)
    # Format output for readability
    pd.set_option('display.float_format', '{:.3f}'.format)
    print("Boundary Enclosing Crystal Atoms (Sorted by Z):")
    print(df.to_string(index=False))


# 5. Plotting (X vs Z side-projection)
fig, ax = plt.subplots(figsize=(8, 6))

# Plot Crystal Bulk (all crystal atoms)
crystal_mask = np.isin(np.arange(len(symbols)), crystal_indices)
ax.scatter(positions[crystal_mask, 0], positions[crystal_mask, 2], s=10, color='gray', label='TiO2 Crystal Slab')

# Plot Water Molecules
water_mask = np.isin(np.arange(len(symbols)), water_indices)
ax.scatter(positions[water_mask, 0], positions[water_mask, 2], s=30, color='blue', label='Water Molecules')

# Plot First Boundary Enclosing Atoms (highlighted)
boundary_mask = np.isin(np.arange(len(symbols)), first_boundary_indices)
ax.scatter(positions[boundary_mask, 0], positions[boundary_mask, 2], s=80, color='red', marker='o', edgecolors='black', linewidths=1.5, label='First Boundary Layer')


ax.set_xlabel('X Position (Å)')
ax.set_ylabel('Z Position (Å)')
ax.set_title('X-Z Projection of TiO2 Interface with Adsorbed Water')
ax.legend(loc='best')
ax.grid(True, linestyle='--', alpha=0.6)

plt.savefig('boundary_enclosing_atoms.png', dpi=300, bbox_inches='tight')
plt.close()
'''

data_mount = os.path.abspath(r"public/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz")
result = run_python_in_sandbox(code, data_mount_path=data_mount)
print("=== DOCKER SANDBOX OUTPUT ===")
print("Exit Code:", result.get("exit_code"))
print("Logs:\n", result.get("logs"))
