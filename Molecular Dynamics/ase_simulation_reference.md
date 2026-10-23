# ASE Molecular Simulations Reference

This document provides quick reference access to ASE functions for Molecular Dynamics and structural analysis in this workspace.

See comprehensive documentation in [ase_molecular_simulations.md](file:///c:/Users/harsh/Chatbox_for_analysis_AI/ChatBot/docs/ase_molecular_simulations.md).

## Key Function Quick Reference

### 1. Structure & Trajectory I/O
- `atoms = read(file_path, index=0); atoms.set_pbc(True)`: single frame Atoms instance.
- `traj = read(file_path, index=':')`: list of Atoms frames. Unpack with `frames = traj if isinstance(traj, list) else [traj]`.
- `with Trajectory('file.traj', 'r') as traj:` memory-efficient streaming for large MD runs.

### 2. Geometry & PBC
- `atoms.get_positions()`: shape (N, 3) in Angstroms.
- `atoms.get_distance(i, j, mic=True)`: minimum image convention distance.
- `atoms.get_all_distances(mic=True)`: (N, N) distance matrix under PBC.
- `atoms.get_scaled_positions(wrap=True)`: fractional coordinates in [0, 1].

### 3. Neighbor Finding
- `nl_i, nl_j, nl_d = neighbor_list('ijd', atoms, cutoff=3.0)`: central atom index, neighbor index, and pair distance.
- `i_oh, j_oh = neighbor_list('ij', atoms, cutoff={('O', 'H'): 1.25})`: selective pair cutoffs.

### 4. Molecular Dynamics & Integrators
- `Langevin(atoms, timestep=1.0*units.fs, temperature_K=300, friction=0.01/units.fs)`
- `VelocityVerlet(atoms, timestep=1.0*units.fs)`
- `MaxwellBoltzmannDistribution(atoms, temperature_K=300.0)`
- `Stationary(atoms)`, `ZeroRotation(atoms)`
