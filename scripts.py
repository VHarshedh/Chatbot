import numpy as np
import os
from ase.io import read
from ase.neighborlist import neighbor_list
import matplotlib.pyplot as plt

# --- Configuration ---
FILE_PATH = 'public/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz'
OUTPUT_PLOT_PATH = 'anatase_tio2_water_analysis.png'
Z_AXIS_INDEX = 2  # Index for the z-coordinate

def analyze_trajectory(file_path: str):
    """
    Loads an ASE trajectory, performs interfacial analysis, and calculates
    structural metrics.
    """
    print(f"Loading trajectory from: {file_path}")
    
    try:
        # 1. Load the trajectory file (returns a list of Atoms)
        traj = read(file_path, index=':')
        
        if not traj:
            print("Error: Trajectory file is empty.")
            return

        print(f"Successfully loaded {len(traj)} frames.")
        
        all_energies = []
        z_positions = []
        water_density_profile = []
        
        # 2. Iterate through frames for analysis
        for i, atoms in enumerate(traj):
            # --- Structural Metrics & Energy ---
            # Calculate potential energy (requires the structure to be valid)
            try:
                potential_energy = atoms.get_potential_energy()
                all_energies.append(potential_energy)
            except Exception as e:
                print(f"Warning: Could not calculate potential energy for frame {i}: {e}")
                continue

            # Extract positions
            positions = atoms.get_positions()
            z_coords = positions[:, Z_AXIS_INDEX]
            z_positions.append(z_coords)

            # --- Interfacial Analysis (Neighbor Search) ---
            # Identify O-H interactions using neighbor_list
            # We search for neighbors of Oxygen atoms, looking for Hydrogen neighbors
            
            oxygen_indices = np.where(np.all(np.isclose(atoms.get_chemical_symbols(), 'O'), axis=1))[0]
            
            for idx in oxygen_indices:
                oxygen_symbol = atoms.get_chemical_symbols()[idx]
                
                # Check neighbors for H atoms
                # Note: neighbor_list requires specific atom types if using advanced filtering, 
                # but we use the general 'ijd' format as per context.
                try:
                    # Find neighbors of this oxygen atom
                    # The result is (i, j, d) indices relative to the current cell
                    n_indices = neighbor_list('ijd', atoms, cutoff=1.5) 
                    
                    for n_idx in n_indices:
                        if n_idx != idx:
                            neighbor_symbol = atoms.get_chemical_symbols()[n_idx]
                            if neighbor_symbol == 'H':
                                # Found an O-H pair
                                print(f"Frame {i}: O-H bond found between index {idx} and {n_idx}")

                except Exception as e:
                    # This might fail if the structure is too small or complex for neighbor search
                    pass


            # --- Water Density Profile (Along Z) ---
            # Calculate density profile based on z-coordinates
            if len(z_coords) > 0:
                # Simple binning for density estimation along the slab
                z_min = np.min(z_coords)
                z_max = np.max(z_coords)
                
                # Create bins for density profile
                z_bins = np.linspace(z_min, z_max, 50)
                density_profile = []
                
                for z in z_bins:
                    # Count atoms within a narrow slice around z
                    count = np.sum(np.abs(z_coords - z) < 0.5)
                    density_profile.append(count / (len(traj) * 1.0)) # Normalized density estimate

                water_density_profile.append(density_profile)


        # --- Results Aggregation ---
        if not all_energies:
            print("No valid energy data collected.")
            return

        avg_energy = np.mean(all_energies)
        std_energy = np.std(all_energies)
        
        # Calculate average density profile across all frames
        mean_density_profile = np.array(water_density_profile)
        
        print("\n--- Analysis Summary ---")
        print(f"Average Potential Energy over trajectory: {avg_energy:.2f} Hartree")
        print(f"Standard Deviation of Potential Energy: {std_energy:.2f} Hartree")
        print(f"Water Density Profile (First Frame Sample): {mean_density_profile[0]}")


        # --- Visualization ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Potential Energy vs. Frame
        axes[0].plot(all_energies, marker='o', linestyle='-', label='Potential Energy')
        axes[0].set_title('Potential Energy Evolution')
        axes[0].set_xlabel('Trajectory Frame')
        axes[0].set_ylabel('Energy (Hartree)')
        axes[0].grid(True)

        # Plot 2: Water Density Profile
        axes[1].plot(z_bins, mean_density_profile[0], label='Density Profile (Frame 0)', color='blue')
        axes[1].set_title('Water Density Profile along Z-axis')
        axes[1].set_xlabel('Z Coordinate (Å)')
        axes[1].set_ylabel('Estimated Density')
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nAnalysis complete. Plot saved to {OUTPUT_PLOT_PATH}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please ensure the file exists in the sandbox.")
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}")

if __name__ == "__main__":
    analyze_trajectory(FILE_PATH)