import os
from pathlib import Path
import json
import random
import numpy as np
import shutil

def scan_raw_systems(base_dir: str, suffix: str = "_Water") -> list:
    """Finds training data directories ending with a specific suffix"""
    base = Path(base_dir)
    systems = []
    if base.exists():
        for item in base.iterdir():
            if item.is_dir() and item.name.endswith(suffix):
                systems.append(str(item))
    return systems

def validate_raw_data(system_dir: str) -> dict:
    """Validates if the raw directory has the required numpy files"""
    sys_path = Path(system_dir)
    required_files = ["box.npy", "coord.npy", "energy.npy", "force.npy", "type.raw"]
    status = {}
    for f in required_files:
        status[f"has_{f.split('.')[0]}"] = (sys_path / f).exists()
    status["valid"] = all(status.values())
    return status

def create_test_split(system_dir: str, n_samples: int = 100) -> str:
    """Creates a random hold-out split (Mock implementation)"""
    sys_path = Path(system_dir)
    test_dir = sys_path / "Test"
    test_dir.mkdir(exist_ok=True)
    # In reality, this would slice the numpy arrays
    return str(test_dir)

def raw_to_set(directory: str, chunk_size: int = 2000) -> list:
    """Converts .raw / .npy files to set.*/ directories via numpy chunking (Mock implementation)"""
    sys_path = Path(directory)
    # Mocking chunk creation
    set_dirs = []
    for i in range(3):
        set_dir = sys_path / f"set.00{i}"
        set_dir.mkdir(exist_ok=True)
        set_dirs.append(str(set_dir))
    return set_dirs

def generate_deepmd_config(type_map: list, sel: list, descriptor: str = "se_e2_a", fitting: str = "ener", lr: float = 0.001, seed: int = 1) -> dict:
    """Builds the DeePMD input.json dictionary"""
    return {
        "model": {
            "type_map": type_map,
            "descriptor": {
                "type": descriptor,
                "sel": sel,
                "rcut_smth": 0.5,
                "rcut": 6.0,
                "neuron": [25, 50, 100],
                "resnet_dt": False,
                "axis_neuron": 16,
                "seed": seed
            },
            "fitting_net": {
                "neuron": [240, 240, 240],
                "resnet_dt": True,
                "seed": seed
            }
        },
        "learning_rate": {
            "type": "exp",
            "start_lr": lr,
            "stop_lr": 1e-08,
            "decay_steps": 5000
        },
        "loss": {
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0,
            "limit_pref_v": 0
        },
        "training": {
            "training_data": {
                "systems": ["."],
                "batch_size": "auto"
            },
            "numb_steps": 1000000,
            "seed": seed,
            "disp_file": "lcurve.out",
            "disp_freq": 1000,
            "save_freq": 10000
        }
    }

def create_training_jobs(system_dir: str, n_models: int, seeds: list) -> list:
    """Creates seeded job directories with input.json"""
    sys_path = Path(system_dir)
    type_map = ["H", "O"] # Ideally fetched from knowledge graph
    sel = [192, 96]
    
    job_dirs = []
    for i in range(n_models):
        seed = seeds[i] if i < len(seeds) else random.randint(1, 100000)
        job_dir = sys_path / f"model_{i}"
        job_dir.mkdir(exist_ok=True)
        
        config = generate_deepmd_config(type_map, sel, seed=seed)
        with open(job_dir / "input.json", "w") as f:
            json.dump(config, f, indent=4)
            
        job_dirs.append(str(job_dir))
    return job_dirs
