import os
from pathlib import Path

def validate_frozen_graphs(graph_dir: str, n_graphs: int = 4) -> list:
    """Resolves and validates frozen .pb graphs"""
    g_dir = Path(graph_dir)
    graphs = []
    if g_dir.exists():
        for i in range(n_graphs):
            pb = g_dir / f"model_{i}.pb"
            if pb.exists():
                graphs.append(str(pb))
    return graphs

def generate_lammps_input(style: str, system_params: dict, batch_num: int, graphs: list) -> str:
    """Generates in.lammps script for DeePMD MD"""
    temp = system_params.get("temp", 300)
    steps = system_params.get("steps", 500000)
    
    inp = f"# LAMMPS Input - Batch {batch_num}\n"
    inp += "units metal\n"
    inp += "boundary p p p\n"
    inp += "atom_style atomic\n\n"
    
    if batch_num == 1:
        inp += "read_data start.data\n"
        inp += f"velocity all create {temp} 12345 mom yes rot yes\n"
    else:
        inp += f"read_restart restart.{batch_num-1}.*\n"
        
    graph_str = " ".join(graphs)
    inp += f"\npair_style deepmd {graph_str} out_freq 100 out_file model_devi.out\n"
    inp += "pair_coeff * *\n\n"
    
    inp += f"fix 1 all nvt temp {temp} {temp} 0.1\n"
    inp += "dump 1 all custom 1000 trajectory.dump id type x y z\n"
    inp += "thermo 100\n"
    
    inp += f"restart {steps} restart.{batch_num}\n"
    inp += f"run {steps}\n"
    
    return inp

def create_batch_chain(output_dir: str, n_batches: int, system_params: dict, graphs: list) -> list:
    """Creates a sequence of MD batches"""
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    batch_dirs = []
    for i in range(1, n_batches + 1):
        b_dir = out_path / f"batch_{i}"
        b_dir.mkdir(exist_ok=True)
        batch_dirs.append(str(b_dir))
        
        inp = generate_lammps_input("deepmd", system_params, i, graphs)
        with open(b_dir / "in.lammps", "w") as f:
            f.write(inp)
            
    return batch_dirs

def generate_submit_chain_script(batch_dirs: list) -> str:
    """Generates submit_from.sh script for sequential slurm execution"""
    script = "#!/bin/bash\n"
    script += "prev_job=''\n"
    for i, b in enumerate(batch_dirs):
        script += f"cd {b}\n"
        if i == 0:
            script += "prev_job=$(sbatch --parsable lammps.sub)\n"
        else:
            script += "prev_job=$(sbatch --parsable --dependency=afterok:$prev_job lammps.sub)\n"
        script += "cd ..\n"
    return script

def check_md_status(batch_dir: str) -> dict:
    """Checks for restart files and errors"""
    b_path = Path(batch_dir)
    restarts = list(b_path.glob("restart.*"))
    return {
        "completed": len(restarts) > 0,
        "restart_files": [str(r) for r in restarts]
    }
