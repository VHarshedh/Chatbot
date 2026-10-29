import os
import sys

code = r'''import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.io import read
from ase.neighborlist import neighbor_list

# --- 1. Load Structure ---
file_path = "/mnt/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
candidate_paths = [
    file_path,
    "public/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz",
    "ChatBot/public/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
]

loaded_path = None
for p in candidate_paths:
    if os.path.exists(p):
        loaded_path = p
        break

if not loaded_path:
    raise FileNotFoundError("Structure file not found at expected paths.")

print(f"Loading structure from: {loaded_path}")
atoms = read(loaded_path, index=0)
atoms.set_pbc(True)
positions = atoms.get_positions()
symbols = np.array(atoms.get_chemical_symbols())
N_total = len(positions)

# --- 2. Separate Water Phase from TiO2 Crystal Slab ---
# Identify water molecules by O-H covalent bonds (cutoff <= 1.25 Å)
i_oh, j_oh = neighbor_list('ij', atoms, {('O', 'H'): 1.25})
water_indices = set(np.where(symbols == 'H')[0])
for idx in i_oh:
    if symbols[idx] == 'O':
        water_indices.add(idx)

crystal_indices = [i for i in range(N_total) if i not in water_indices]
water_indices = sorted(list(water_indices))

water_z = positions[water_indices, 2]
z_w_min = water_z.min()
z_w_max = water_z.max()

# Separate crystal into bottom slab and top slab relative to the water channel
bot_crystal_idx = [i for i in crystal_indices if positions[i, 2] < z_w_min]
top_crystal_idx = [i for i in crystal_indices if positions[i, 2] > z_w_max]

print(f"Total Atoms: {N_total}")
print(f"Water Molecules: {len(water_indices) // 3} ({len(water_indices)} atoms: {np.sum(symbols[water_indices] == 'H')} H, {np.sum(symbols[water_indices] == 'O')} O)")
print(f"TiO2 Crystal Slab Atoms: {len(crystal_indices)} ({np.sum(symbols[crystal_indices] == 'Ti')} Ti, {np.sum(symbols[crystal_indices] == 'O')} O)")
print(f"Water Z-envelope: {z_w_min:.2f} Å to {z_w_max:.2f} Å\n")

# --- 3. Isolate the 1st Plane Alone (Top and Bottom Enclosing Boundaries) ---
# Bottom slab: outermost surface atoms facing water have maximum Z in bottom slab
max_z_bot = max(positions[i, 2] for i in bot_crystal_idx)
# Top slab: outermost surface atoms facing water have minimum Z in top slab
min_z_top = min(positions[i, 2] for i in top_crystal_idx)

# Plane tolerance: 0.35 Å isolates strictly the 1st atomic plane (2nd plane is > 0.46 Å away)
plane_tol = 0.35
bot_p1_indices = [i for i in bot_crystal_idx if positions[i, 2] >= (max_z_bot - plane_tol)]
top_p1_indices = [i for i in top_crystal_idx if positions[i, 2] <= (min_z_top + plane_tol)]

# Calculate minimum distance to nearest water atom using MIC
nl_i, nl_j, nl_d = neighbor_list('ijd', atoms, cutoff=4.0)
c_set = set(bot_p1_indices + top_p1_indices)
w_set = set(water_indices)
min_dists = {}
for i, j, d in zip(nl_i, nl_j, nl_d):
    if i in c_set and j in w_set:
        if i not in min_dists or d < min_dists[i]:
            min_dists[i] = d

# Build boundary atoms data
boundary_rows = []
for i in sorted(bot_p1_indices, key=lambda idx: positions[idx, 2]):
    boundary_rows.append({
        'Index': i,
        'Symbol': symbols[i],
        'Layer': 'Bottom Enclosing Plane',
        'X': positions[i, 0],
        'Y': positions[i, 1],
        'Z': positions[i, 2],
        'Min_Dist_to_Water': min_dists.get(i, np.nan)
    })

for i in sorted(top_p1_indices, key=lambda idx: positions[idx, 2]):
    boundary_rows.append({
        'Index': i,
        'Symbol': symbols[i],
        'Layer': 'Top Enclosing Plane',
        'X': positions[i, 0],
        'Y': positions[i, 1],
        'Z': positions[i, 2],
        'Min_Dist_to_Water': min_dists.get(i, np.nan)
    })

df_boundary = pd.DataFrame(boundary_rows)

print("=" * 70)
print("FIRST BOUNDARY LAYER ANALYSIS (1ST PLANE ALONE)")
print("=" * 70)
print(f"Bottom 1st Plane Atoms (Z ~ {max_z_bot:.2f} Å): {len(bot_p1_indices)}")
print(f"Top 1st Plane Atoms    (Z ~ {min_z_top:.2f} Å): {len(top_p1_indices)}")
print(f"Total Enclosing 1st Plane Atoms: {len(boundary_rows)}\n")

pd.set_option('display.float_format', '{:.4f}'.format)
print("Formatted Coordinate Table of 1st Plane Atoms:")
print(df_boundary.to_string(index=False))

# --- 4. Generate 2D Side-View Projection Plot (X vs Z) ---
fig, ax = plt.subplots(figsize=(8, 10))

# Plot full crystal slab in gray
ax.scatter(positions[crystal_indices, 0], positions[crystal_indices, 2],
           c='#888888', s=25, alpha=0.5, label='TiO2 Crystal Slab')

# Plot water phase in sky blue
ax.scatter(positions[water_indices, 0], positions[water_indices, 2],
           c='#3b82f6', s=20, alpha=0.5, label='Water Molecules (H2O)')

# Highlight 1st plane bottom atoms
ax.scatter([r['X'] for r in boundary_rows if 'Bottom' in r['Layer']],
           [r['Z'] for r in boundary_rows if 'Bottom' in r['Layer']],
           c='#ef4444', edgecolors='black', s=90, marker='o', linewidths=1.5,
           label=f'Bottom 1st Plane ({len(bot_p1_indices)} atoms)', zorder=5)

# Highlight 1st plane top atoms
ax.scatter([r['X'] for r in boundary_rows if 'Top' in r['Layer']],
           [r['Z'] for r in boundary_rows if 'Top' in r['Layer']],
           c='#10b981', edgecolors='black', s=90, marker='s', linewidths=1.5,
           label=f'Top 1st Plane ({len(top_p1_indices)} atoms)', zorder=5)

ax.set_xlabel('X Position (Å)', fontsize=12)
ax.set_ylabel('Z Position (Å)', fontsize=12)
ax.set_title('Side-View Projection: 1st Plane Atoms Enclosing Water', fontsize=13, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='upper right', framealpha=0.9)

output_fig = 'boundary_enclosing_atoms.png'
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nPlot saved successfully as: {output_fig}")
'''

# --- Execution Driver (Docker Sandbox with Local Fallback) ---
try:
    from backend.tools.sandbox_tools import run_python_in_sandbox
    data_mount = os.path.abspath(r"public/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz")
    if not os.path.exists(data_mount):
        data_mount = os.path.abspath(r"ChatBot/public/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz")
    result = run_python_in_sandbox(code, data_mount_path=data_mount)
    print("=== DOCKER SANDBOX OUTPUT ===")
    print("Exit Code:", result.get("exit_code"))
    logs = result.get("logs") or result.get("stdout") or result.get("stderr")
    print("Logs:\n", logs)
    if result.get("exit_code") != 0:
        print("\n=== RUNNING DIRECTLY ON HOST (SANDBOX FALLBACK) ===")
        exec(code)
except Exception:
    exec(code)
