"""Shared agent access to the installed scientific tool suite."""
import importlib
import inspect
import json
from pathlib import Path
import re

MODULES = ('analysis_tools', 'dft_tools', 'md_tools', 'training_tools', 'slurm_tools')


def scientific_tool_catalog():
    lines = ['Scientific helpers are importable inside the sandbox from backend.tools.<module>.']
    for name in MODULES:
        module = importlib.import_module('backend.tools.' + name)
        functions = [f'{key}{inspect.signature(value)}' for key, value in vars(module).items()
                     if inspect.isfunction(value) and value.__module__ == module.__name__ and not key.startswith('_')]
        lines.append(name + ': ' + '; '.join(functions))
    lines.append('SLURM helpers require a configured cluster; sandbox networking remains disabled. Never invent job IDs.')
    return '\n'.join(lines)


def resolve_structure_paths(prompt):
    root = Path(__file__).resolve().parents[2]
    paths = []
    # Resolve explicit files only within project data locations.
    for folder in ('public', 'data', 'DFT', 'Analysis'):
        base = root/folder
        if not base.exists():
            continue
        for path in base.iterdir():
            if path.is_file() and path.suffix.lower() in {'.extxyz', '.xyz', '.cif', '.lmp', '.traj'}:
                p_lower = prompt.lower()
                if path.name.lower() in p_lower:
                    paths.append(str(path))
                elif re.search(rf"(?<![a-zA-Z0-9_]){re.escape(path.stem.lower())}(?![a-zA-Z0-9_])", p_lower):
                    paths.append(str(path))
    paths = list(dict.fromkeys(paths))
    if len(paths) > 1:
        has_water = bool(re.search(r'\b(water|h2o|molecular|adsorb\w*|interfac\w*)\b', prompt, re.I))
        disambiguated = [p for p in paths if ("water" in Path(p).name.lower()) == has_water]
        if disambiguated:
            paths = disambiguated

    if not paths and re.search(r'\banatase\b|\btio2\b', prompt, re.I):
        if re.search(r'\b(water|h2o|molecular|adsorb\w*|interfac\w*)\b', prompt, re.I):
            water_path = root/'public'/'Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz'
            if water_path.exists():
                paths.append(str(water_path))
        else:
            pure_path = root/'public'/'anatase_tio2.extxyz'
            if pure_path.exists():
                paths.append(str(pure_path))
            else:
                water_path = root/'public'/'Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz'
                if water_path.exists():
                    paths.append(str(water_path))
    return paths


def structure_context(prompt):
    from backend.tools.analysis_tools import inspect_structure_metadata
    cards = []
    for path in resolve_structure_paths(prompt):
        card = inspect_structure_metadata(path)
        card['sandbox_path'] = '/mnt/data/' + Path(path).name
        cards.append(card)
    return '<inspected_structure_metadata>\n' + json.dumps(cards) + '\n</inspected_structure_metadata>' if cards else ''
