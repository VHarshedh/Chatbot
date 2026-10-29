"""Static checks and evidence-based output auditing (not a scientific proof)."""
import ast
import importlib
import inspect
from pathlib import Path
import re
import warnings


def validate_scientific_script_ast(code, user_prompt=""):
    issues = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tree = ast.parse(code)
        issues.extend(str(w.message) for w in caught)
    except SyntaxError as exc:
        return {"valid": False, "issues": [f"SyntaxError at line {exc.lineno}: {exc.msg}"]}
    tuple_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            name = ast.unparse(call.func)
            if name.endswith("neighbor_list") and call.args and isinstance(call.args[0], ast.Constant) and len(str(call.args[0].value)) > 1:
                tuple_names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name in {"atoms.get_neighbors", "atom.get_neighbors", "frame.get_neighbors"}:
                issues.append("ASE Atoms has no get_neighbors(); use ase.neighborlist.neighbor_list.")
            if name in {"plt.show", "pyplot.show"}:
                issues.append("Headless plotting requires savefig() and close(), not show().")
            if name.endswith("scatter") and len(node.args) >= 2 and re.search(r"side[- ]view|x-z|\bxz\b|x vs z", user_prompt, re.I):
                args = [ast.unparse(a).replace(" ", "") for a in node.args[:2]]
                if args[0].endswith("[:,0]") and args[1].endswith("[:,1]"):
                    issues.append("Requested X-Z projection uses X-Y; plot column 2 as vertical axis.")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and '\text{' in node.value:
            issues.append("Tab escape corrupted a title; use Unicode text or a raw string.")
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in tuple_names and isinstance(node.slice, ast.Tuple):
            issues.append(f"{node.value.id} is a tuple of neighbor arrays; unpack before multidimensional indexing.")
    return {"valid": not issues, "issues": issues}


def inspect_local_python_api(symbol_path):
    parts = symbol_path.strip().split(".")
    if parts[0] not in {"ase", "numpy", "scipy"} or any(p.startswith("_") for p in parts):
        return "Unsupported symbol; use a public ASE, NumPy or SciPy API."
    for end in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:end]))
            for part in parts[end:]:
                obj = getattr(obj, part)
            try:
                signature = str(inspect.signature(obj)) if callable(obj) else ""
            except (ValueError, TypeError):
                signature = " (signature unavailable)"
            return f"Symbol: {symbol_path}{signature}\nInstalled host API; sandbox version may differ.\n{(inspect.getdoc(obj) or 'No docstring available.')[:1800]}"
        except (ImportError, AttributeError):
            continue
    return f"Could not resolve symbol '{symbol_path}'."


def local_error_context(code, stderr):
    symbols = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in {'ase', 'numpy', 'scipy'}:
                symbols.extend(node.module + '.' + alias.name for alias in node.names)
    except SyntaxError:
        pass
    if 'get_neighbors' in stderr or 'neighbor_list' in code:
        symbols.insert(0, 'ase.neighborlist.neighbor_list')
    if 'get_all_distances' in code:
        symbols.insert(0, 'ase.Atoms.get_all_distances')
    return '\n\n'.join(inspect_local_python_api(s) for s in dict.fromkeys(symbols[:3]))


def artifact_metadata(paths):
    result = []
    for item in paths:
        path = Path(item)
        meta = {'path': str(path), 'name': path.name, 'size_bytes': path.stat().st_size if path.is_file() else 0}
        meta['nonempty'] = meta['size_bytes'] > 0
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
            try:
                from PIL import Image
                with Image.open(path) as img:
                    meta['width'], meta['height'] = img.size
                    img.verify()
                meta['valid_image'] = True
            except Exception as exc:
                meta.update(valid_image=False, error=str(exc))
        result.append(meta)
    return result


def audit_sandbox_execution_result(stdout, stderr, artifacts, user_prompt='', structure_metadata=None):
    issues = []
    if re.search(r'(?im)^\s*Error:|no boundary atoms|no crystal-water interactions|skipping plot generation', stdout):
        issues.append('Output reports a failed or empty scientific result.')
    if re.search(r'SyntaxWarning|invalid escape sequence', stderr):
        issues.append('Python reported invalid string escapes.')
    metas = artifacts if artifacts and isinstance(artifacts[0], dict) else artifact_metadata(artifacts)
    requested = set(re.findall(r'[\w.-]+\.(?:png|jpg|jpeg|svg|pdf)', user_prompt, re.I))
    if requested - {m['name'] for m in metas}:
        issues.append('Missing requested artifacts: ' + ', '.join(sorted(requested - {m['name'] for m in metas})))
    if re.search(r'\b(plot|figure|visualize|visualise)\b', user_prompt, re.I) and not any(Path(m['name']).suffix.lower() in {'.png', '.jpg', '.jpeg', '.svg', '.pdf'} for m in metas):
        issues.append('A plot was requested but no image artifact was generated.')
    for meta in metas:
        if meta['name'] in requested and Path(meta['name']).suffix.lower() in {'.png', '.jpg', '.jpeg'} and meta.get('size_bytes', 0) < 5120:
            issues.append('Requested plot is suspiciously small (less than 5 KB): ' + meta['name'])
        if not meta.get('nonempty', meta.get('size_bytes', 0) > 0) or meta.get('valid_image') is False:
            issues.append('Empty or corrupt artifact: ' + meta['name'])
    count = re.search(r'(?:boundary(?: crystal)? atoms?\s*[:=]\s*)(\d+)', stdout, re.I)
    crystal = (structure_metadata or {}).get('phase_partition_preview', {}).get('crystal_slab_atom_count')
    if count and (int(count[1]) == 0 or (crystal is not None and int(count[1]) > crystal)):
        issues.append('Boundary atom count is inconsistent with the inspected crystal partition.')
    return {'valid': not issues, 'issues': issues, 'artifact_metadata': metas,
            'scope': 'Checks observable failures and artifact integrity; does not certify scientific correctness or plot content.'}
