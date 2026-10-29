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

def _load_raw(directory):
    root = Path(directory)
    types = np.loadtxt(root / "type.raw", ndmin=1)
    if not len(types) or np.any(types < 0) or np.any(types != types.astype(int)):
        raise ValueError("type.raw must contain nonnegative integer atom types")
    arrays = {name: np.loadtxt(root / (name + ".raw"), ndmin=2) for name in ("box", "coord", "energy", "force")}
    if (root / "virial.raw").exists():
        arrays["virial"] = np.loadtxt(root / "virial.raw", ndmin=2)
    frames = arrays["coord"].shape[0]
    widths = {"box": 9, "coord": 3*len(types), "force": 3*len(types), "energy": 1, "virial": 9}
    for name, array in arrays.items():
        if array.shape != (frames, widths[name]) or not np.all(np.isfinite(array)):
            raise ValueError(f"Invalid {name}.raw shape or nonfinite values: {array.shape}")
    return types.astype(int), arrays


def validate_raw_data(system_dir):
    status = {"has_" + key: (Path(system_dir)/(key+".raw")).exists() for key in ("box", "coord", "energy", "force", "type")}
    try:
        types, arrays = _load_raw(system_dir)
        status.update(valid=True, n_atoms=len(types), n_frames=len(arrays["coord"]))
    except (OSError, ValueError) as exc:
        status.update(valid=False, error=str(exc))
    return status


def create_test_split(system_dir, n_samples=100, seed=1):
    """Write disjoint Train/Test raw systems, retaining untouched original data."""
    root = Path(system_dir)
    types, arrays = _load_raw(root)
    count = len(arrays["coord"])
    if not 0 < n_samples < count:
        raise ValueError("n_samples must leave at least one training frame")
    if any((root/name).exists() for name in ("Train", "Test")):
        raise ValueError("Train/Test already exists; choose a fresh output location")
    indices = np.random.default_rng(seed).permutation(count)
    for name, selected in (("Test", indices[:n_samples]), ("Train", indices[n_samples:])):
        dest = root/name
        dest.mkdir(exist_ok=False)
        np.savetxt(dest/"type.raw", types, fmt="%d")
        np.savetxt(dest/"frame_indices.txt", selected, fmt="%d")
        if (root/"type_map.raw").exists():
            shutil.copy2(root/"type_map.raw", dest/"type_map.raw")
        for key, array in arrays.items():
            np.savetxt(dest/(key+".raw"), array[selected])
    return str(root/"Test")


def raw_to_set(directory, chunk_size=2000):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    root = Path(directory)
    _, arrays = _load_raw(root)
    if list(root.glob("set.*")):
        raise ValueError("Existing set directories must be handled before conversion")
    result = []
    for index, start in enumerate(range(0, len(arrays["coord"]), chunk_size)):
        dest = root/f"set.{index:03d}"
        dest.mkdir()
        for key, array in arrays.items():
            np.save(dest/(key+".npy"), array[start:start+chunk_size])
        result.append(str(dest))
    return result


def generate_deepmd_config(type_map: list, sel: list, descriptor: str = "se_e2_a", fitting: str = "ener", lr: float = 0.005, seed: int = 1) -> dict:
    """Builds the DeePMD input.json dictionary"""
    return {
        "model": {
            "type_map": type_map,
            "descriptor": {
                "type": descriptor,
                "sel": sel,
                "rcut_smth": 3.0,
                "rcut": 6.0,
                "neuron": [120, 120, 120],
                "resnet_dt": False,
                "axis_neuron": 12,
                "seed": seed
            },
            "fitting_net": {
                "neuron": [120, 120, 120],
                "resnet_dt": True,
                "seed": seed
            }
        },
        "learning_rate": {
            "type": "exp",
            "start_lr": lr,
            "decay_rate": 0.95,
            "decay_steps": 5000
        },
        "loss": {
            "type": fitting,
            "start_pref_e": 0.02,
            "limit_pref_e": 8,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0,
            "limit_pref_v": 0
        },
        "training": {
            "training_data": {
                "systems": ["./Train"],
                "batch_size": "auto"
            },
            "validation_data": {"systems": ["./Test"], "batch_size": "auto"},
            "numb_steps": 1000000,
            "seed": seed,
            "disp_file": "lcurve.out",
            "disp_freq": 100,
            "save_freq": 1000
        }
    }

def create_training_jobs(system_dir: str, n_models: int, seeds: list) -> list:
    """Creates seeded job directories with input.json"""
    sys_path = Path(system_dir)
    type_map = ["H", "O"] # Ideally fetched from knowledge graph
    sel = [120, 64]
    
    job_dirs = []
    for i in range(n_models):
        seed = seeds[i] if i < len(seeds) else random.randint(1, 100000)
        job_dir = sys_path / f"model_{i}"
        job_dir.mkdir(exist_ok=True)
        
        config = generate_deepmd_config(type_map, sel, seed=seed)
        config["training"]["training_data"]["systems"] = ["../Train"]
        config["training"]["validation_data"]["systems"] = ["../Test"]
        with open(job_dir / "input.json", "w") as f:
            json.dump(config, f, indent=4)
            
        job_dirs.append(str(job_dir))
    return job_dirs
