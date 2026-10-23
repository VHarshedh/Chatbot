# ASE (Atomic Simulation Environment) Guide for Molecular Simulations

This reference guide documents canonical ASE functions, idioms, parameter signatures, and best practices extensively used in atomic and molecular simulations, surface slab analysis, neighbor searches, and trajectory workflows.

---

## 1. File Input/Output & Trajectory Management (`ase.io`)

### 1.1 Reading Structures and Trajectories (`ase.io.read`)
The `read` function is the primary entry point for loading atomic configurations from external files (e.g. Extended XYZ `.extxyz`, CIF `.cif`, LAMMPS data, VASP POSCAR/OUTCAR, Quantum ESPRESSO `.in`/`.out`).

```python
from ase.io import read

# Load a single frame (index=0 returns an ase.Atoms instance)
atoms = read("structure.extxyz", index=0)
atoms.set_pbc(True)

# Load the final converged/relaxed frame
final_atoms = read("relaxation.extxyz", index=-1)

# CRITICAL SYNTAX: Load entire trajectory as a list of Atoms (index=':')
# Returns list[ase.Atoms]. A Python list has NO .get_positions() attribute!
traj = read("trajectory.extxyz", index=":")
frames = traj if isinstance(traj, list) else [traj]
first_frame = frames[0]
positions_first_frame = first_frame.get_positions()

# Strided reading (e.g., sample every 10th frame)
subsampled_traj = read("trajectory.extxyz", index="::10")
```

### 1.2 Memory-Efficient Streaming Trajectories (`ase.io.trajectory.Trajectory`)
For large molecular dynamics trajectories that exceed available RAM, use `Trajectory` to stream frames on-demand without loading the full file into memory:

```python
from ase.io.trajectory import Trajectory

# Iterate through frames on disk with minimal memory overhead
with Trajectory("large_md_simulation.traj", "r") as traj:
    total_frames = len(traj)
    for step, frame in enumerate(traj):
        epot = frame.get_potential_energy()
        temp = frame.get_temperature()
        # Analyze frame coordinates
        z_coords = frame.get_positions()[:, 2]
```

### 1.3 Writing Trajectories and Configurations (`ase.io.write`)
```python
from ase.io import write

# Save a single Atoms structure
write("optimized_slab.extxyz", atoms, format="extxyz")

# Append a frame to an existing trajectory
write("md_run.extxyz", atoms, format="extxyz", append=True)

# Write a list of Atoms frames into a single trajectory file
write("full_sampled_run.extxyz", frames, format="extxyz")

# Export to LAMMPS data format with specific atom style
write("structure.data", atoms, format="lammps-data", atom_style="atomic")
```

---

## 2. Core `ase.Atoms` Methods and Properties

The `Atoms` class represents a collection of atoms in 3D space with an optional periodic unit cell.

### 2.1 Initialization & Basic Attributes
```python
import numpy as np
from ase import Atoms

# Create an Atoms object from scratch
cell_box = [10.23, 10.23, 35.0]  # Or 3x3 matrix
atoms = Atoms(
    symbols=["Ti", "O", "O"],
    positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 0.0], [-1.5, -1.5, 0.0]],
    cell=cell_box,
    pbc=[True, True, True]
)

# Number of atoms
N = len(atoms)

# Chemical symbols and atomic numbers
symbols = np.array(atoms.get_chemical_symbols())
numbers = atoms.get_atomic_numbers()
formula = atoms.get_chemical_formula()
```

### 2.2 Coordinates, Cell, and Periodic Boundary Conditions (PBC)
```python
# Absolute Cartesian positions (shape: (N, 3) in Angstroms)
positions = atoms.get_positions()
atoms.set_positions(positions + np.array([0.0, 0.0, 1.0]))

# Fractional / Scaled coordinates (shape: (N, 3), range [0, 1])
scaled_pos = atoms.get_scaled_positions(wrap=True)
atoms.set_scaled_positions(scaled_pos)

# Unit Cell vectors and dimensions
cell = atoms.get_cell()            # 3x3 Cell object
lengths, angles = atoms.cell.lengths_and_angles() # a, b, c, alpha, beta, gamma
volume = atoms.get_volume()        # Cell volume in Angstroms^3

# Periodic boundary conditions
pbc = atoms.get_pbc()              # array([True, True, True])
atoms.set_pbc([True, True, True])

# Wrap all atoms back into the primary unit cell
atoms.wrap(eps=1e-7)
```

### 2.3 Distances, Angles, and Minimum Image Convention (MIC)
In periodic simulations, distances between atoms across cell boundaries MUST enforce the minimum image convention (`mic=True`):

```python
# Scalar distance between atom i and atom j taking PBC into account
dist_ij = atoms.get_distance(0, 1, mic=True)

# Displacement vector pointing from atom 0 to atom 1 (r_1 - r_0) under MIC
vec_01 = atoms.get_distance(0, 1, mic=True, vector=True)

# Full pairwise distance matrix (N, N) with minimum image convention
all_dists = atoms.get_all_distances(mic=True)

# Bond angle between three atoms (a1 - a2 - a3) in degrees
angle_deg = atoms.get_angle(0, 1, 2, mic=True)

# Dihedral angle between four atoms (a1 - a2 - a3 - a4) in degrees
dihedral_deg = atoms.get_dihedral(0, 1, 2, 3, mic=True)
```

### 2.4 Subsetting, Slicing, and Composition
```python
# Masking: select only Titanium atoms
ti_mask = (symbols == "Ti")
ti_atoms = atoms[ti_mask]  # Returns a new Atoms object

# Slice by index list
sub_atoms = atoms[[0, 4, 12, 18]]

# Delete atoms in-place
del atoms[atoms.positions[:, 2] < 2.0]

# Combine systems (e.g., surface slab + water adsorbate)
combined = slab + water_molecule
```

---

## 3. High-Performance Neighbor Finding (`ase.neighborlist`)

The `neighbor_list` function is the workhorse of structural analysis, coordination number calculation, radial distribution functions, and interfacial boundary identification.

### 3.1 Unpacking Quantities (`'i'`, `'j'`, `'d'`, `'ijd'`)
The first argument specifies the return quantities string:
- `'i'`: Array of central atom indices.
- `'j'`: Array of neighboring atom indices.
- `'d'`: Array of neighbor distances ($r_{ij}$).
- `'D'`: Array of Cartesian displacement vectors ($\vec{r}_j - \vec{r}_i$ under MIC).
- `'S'`: Array of integer unit cell shift vectors.

```python
from ase.neighborlist import neighbor_list, natural_cutoffs

# 1. Unpack (i, j, d) with a scalar distance cutoff (e.g., 3.2 Angstroms)
nl_i, nl_j, nl_d = neighbor_list("ijd", atoms, cutoff=3.2, self_interaction=False)

# 2. Element-pair specific cutoffs (dictionary)
# E.g., covalent O-H bond cutoff vs Ti-O coordination cutoff
cutoffs = {
    ("O", "H"): 1.25,
    ("Ti", "O"): 2.40,
    ("Ti", "Ti"): 3.50
}
i_pairs, j_pairs, dists = neighbor_list("ijd", atoms, cutoff=cutoffs, self_interaction=False)

# 3. Dynamic natural cutoffs based on covalent radii of chemical elements
nat_cuts = natural_cutoffs(atoms, mult=1.1)
i_idx, j_idx = neighbor_list("ij", atoms, cutoff=nat_cuts)
```

### 3.2 Canonical Interfacial Water & Lattice Separation
```python
# Step 1: Identify all water molecules by detecting O-H covalent bonds (cutoff = 1.25 A)
i_oh, j_oh = neighbor_list("ij", atoms, cutoff={("O", "H"): 1.25}, self_interaction=False)
symbols = np.array(atoms.get_chemical_symbols())

water_indices = set(np.where(symbols == "H")[0])
for idx in i_oh:
    if symbols[idx] == "O":
        water_indices.add(idx)

# Step 2: Crystal slab atoms are Ti and O atoms NOT belonging to water
crystal_indices = {
    idx for idx, sym in enumerate(symbols)
    if sym in ("Ti", "O") and idx not in water_indices
}

# Step 3: Find first boundary crystal atoms that enclose the water layer
nl_i, nl_j, nl_d = neighbor_list("ijd", atoms, cutoff=3.2, self_interaction=False)

enclosing_crystal_dict = {}
for i_atom, j_atom, dist in zip(nl_i, nl_j, nl_d):
    if i_atom in crystal_indices and j_atom in water_indices:
        if i_atom not in enclosing_crystal_dict or dist < enclosing_crystal_dict[i_atom]:
            enclosing_crystal_dict[i_atom] = dist

first_boundary_indices = sorted(enclosing_crystal_dict.keys(), key=lambda idx: atoms.positions[idx, 2])
```

---

## 4. Structure Generation & Slab Modeling (`ase.build`)

### 4.1 Surface Slabs
```python
from ase.build import surface, fcc111, fcc100, bcc110

# Create an FCC(111) Pt slab with 4 layers and 12 Angstrom vacuum
pt_slab = fcc111("Pt", size=(4, 4, 4), vacuum=12.0)

# Create an arbitrary Miller index surface slab (e.g. Anatase TiO2 (101))
# bulk_anatase is a relaxed unit cell of TiO2
# anatase_slab = surface(bulk_anatase, indices=(1, 0, 1), layers=4, vacuum=15.0)
```

### 4.2 Supercells and Repetition
```python
from ase.build import make_supercell

# Simple integer repetition of unit cells
supercell = unit_cell * (2, 2, 4)

# Arbitrary transformation matrix
P = [[2, 1, 0], [-1, 2, 0], [0, 0, 1]]
general_supercell = make_supercell(unit_cell, P)
```

### 4.3 Adding Adsorbates
```python
from ase.build import molecule, add_adsorbate

# Load water molecule
h2o = molecule("H2O")

# Add water molecule to Pt slab at height 2.2 Angstroms above ontop site
add_adsorbate(pt_slab, h2o, height=2.2, position="ontop")
```

---

## 5. Structural & Statistical Analysis (`ase.geometry`)

### 5.1 Radial Distribution Function (RDF)
```python
from ase.geometry.rdf import get_rdf

# Compute g(r) for all pairwise interactions or specific element pairs
# rmax in Angstroms, nbins = number of distance bins
rmax = 8.0
nbins = 200
r_vals, g_r = get_rdf(atoms, rmax=rmax, nbins=nbins)
```

### 5.2 Minimum Image Convention Vector Calculation
```python
from ase.geometry import find_mic

# Calculate periodic displacement vectors and scalar distances between arbitrary points
disp_vectors, distances = find_mic(positions_b - positions_a, atoms.get_cell(), atoms.get_pbc())
```

---

## 6. Molecular Dynamics Simulation (`ase.md`)

### 6.1 Temperature Initialization
```python
from ase import units
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

# Initialize atomic velocities from a Maxwell-Boltzmann distribution at T = 300 K
MaxwellBoltzmannDistribution(atoms, temperature_K=300.0)

# Remove net translational momentum (drift)
Stationary(atoms)

# Remove net rigid-body rotation
ZeroRotation(atoms)

# Measure current instantaneous kinetic temperature
current_T = atoms.get_temperature()
```

### 6.2 Integrators (NVE, NVT, NPT)
```python
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.io.trajectory import Trajectory
from ase.md import MDLogger

# --- NVT Langevin Dynamics ---
timestep = 1.0 * units.fs
dyn = Langevin(
    atoms,
    timestep=timestep,
    temperature_K=300.0,
    friction=0.01 / units.fs,  # Friction coefficient gamma
    trajectory="langevin_run.traj"
)

# Attach logger to record thermodynamic properties every 50 steps
dyn.attach(MDLogger(dyn, atoms, "md.log", header=True, mode="w"), interval=50)

# Run simulation for 2000 steps (2 ps)
dyn.run(steps=2000)
```

---

## 7. Structure Optimization & Minimization (`ase.optimize`)

```python
from ase.optimize import BFGS, FIRE, LBFGS

# BFGS quasi-Newton line search optimizer
optimizer = BFGS(atoms, trajectory="relax.traj", logfile="relax.log")

# Relax until maximum force on any atom is below 0.02 eV/Angstrom
optimizer.run(fmax=0.02, steps=150)
```

---

## 8. Constraints (`ase.constraints`)

```python
from ase.constraints import FixAtoms, FixBondLength

# Freeze specific substrate atoms (e.g., bottom 2 layers of a metal slab)
z_threshold = atoms.positions[:, 2].min() + 2.5
freeze_mask = (atoms.positions[:, 2] <= z_threshold)
constraint = FixAtoms(mask=freeze_mask)
atoms.set_constraint(constraint)
```

---

## 9. Physical Units Reference (`ase.units`)

ASE uses atomic units internally:
- Length: Angstroms ($\text{\AA}$)
- Energy: Electron volts ($\text{eV}$)
- Mass: Atomic mass units ($\text{amu}$ or $\text{u}$)
- Time: $\text{\AA} \sqrt{\text{u}/\text{eV}} \approx 10.18\ \text{fs}$
- Conversions:
  - `timestep = 1.0 * ase.units.fs`
  - `pressure = 1.0 * ase.units.bar`
  - `temperature = 300.0` (Kelvin)
  - `force = eV / Angstrom`
