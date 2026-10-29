import os
from ase.io import read
import numpy as np

# Define the file path based on the inspection metadata
file_path = "public/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    try:
        # Load the structure using ase.io.read with format='extxyz'
        atoms = read(file_path, format='extxyz')
        
        print("--- Structure Loaded Successfully via ASE ---")
        
        # Extract forces using the dedicated method
        # The header declares 'force:R:3' (singular), so ASE puts it in atoms.arrays['force']
        forces = atoms.arrays.get('force') if 'force' in atoms.arrays else (atoms.arrays.get('forces') if 'forces' in atoms.arrays else None)
        if forces is None and atoms.calc:
            try:
                forces = atoms.get_forces()
            except Exception:
                forces = None
        
        if forces is not None:
            print("\n--- Force Extraction Successful ---")
            
            # Extract the force vector of the first atom (index 0)
            first_atom_force = forces[0]
            
            print(f"Force vector of the first atom (Index 0):")
            print(first_atom_force)
        else:
            print("\nWarning: Failed to extract forces using atoms.get_forces().")

    except Exception as e:
        print(f"An error occurred during file processing or ASE loading: {e}")