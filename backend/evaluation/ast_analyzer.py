import ast
import builtins
import re
from typing import Dict, Any, List, Set, Optional

class AstAnalyzer:
    """
    Performs modular static AST parsing and semantic code inspection
    for computational chemistry and scientific computing scripts.
    """

    DANGEROUS_CALLS: Set[str] = {
        "os.system", "subprocess.run", "subprocess.Popen", "subprocess.call",
        "shutil.rmtree", "eval", "exec", "__import__"
    }

    SCIENTIFIC_LIBRARIES: Set[str] = {
        "ase", "numpy", "np", "scipy", "matplotlib", "plt", "pandas", "pd",
        "pymatgen", "mdanalysis", "torch", "sklearn"
    }

    def __init__(self, code: str):
        self.code = code or ""
        self.tree: Optional[ast.AST] = None
        self.syntax_valid: bool = False
        self.syntax_error_msg: Optional[str] = None
        self.syntax_error_lineno: Optional[int] = None
        self._parse()

    def _parse(self):
        if not self.code.strip():
            self.syntax_valid = False
            self.syntax_error_msg = "Empty code snippet provided."
            return

        try:
            self.tree = ast.parse(self.code)
            self.syntax_valid = True
        except SyntaxError as e:
            self.syntax_valid = False
            self.syntax_error_msg = e.msg
            self.syntax_error_lineno = e.lineno
        except Exception as e:
            self.syntax_valid = False
            self.syntax_error_msg = str(e)

    def extract_imports(self) -> Dict[str, Set[str]]:
        """Extracts top-level and alias imports."""
        if not self.tree:
            return {"modules": set(), "from_imports": set(), "aliases": set()}

        modules: Set[str] = set()
        from_imports: Set[str] = set()
        aliases: Set[str] = set()

        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name.split('.')[0])
                    if alias.asname:
                        aliases.add(alias.asname)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.add(node.module.split('.')[0])
                    for alias in node.names:
                        from_imports.add(f"{node.module}.{alias.name}")
                        if alias.asname:
                            aliases.add(alias.asname)

        return {
            "modules": modules,
            "from_imports": from_imports,
            "aliases": aliases
        }

    def extract_functions(self) -> List[Dict[str, Any]]:
        """Finds all function definitions, their arguments, and docstrings."""
        if not self.tree:
            return []

        functions = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node)
                args = [a.arg for a in node.args.args]
                functions.append({
                    "name": node.name,
                    "args": args,
                    "docstring": doc,
                    "lineno": node.lineno
                })
        return functions

    def check_safety(self) -> Dict[str, Any]:
        """Detects potentially dangerous system or arbitrary execution calls."""
        violations = []

        if self.tree:
            for node in ast.walk(self.tree):
                if isinstance(node, ast.Call):
                    call_name = ""
                    if isinstance(node.func, ast.Name):
                        call_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        parts = []
                        curr = node.func
                        while isinstance(curr, ast.Attribute):
                            parts.append(curr.attr)
                            curr = curr.value
                        if isinstance(curr, ast.Name):
                            parts.append(curr.id)
                        call_name = ".".join(reversed(parts))

                    if call_name in self.DANGEROUS_CALLS:
                        violations.append(f"Dangerous function invocation detected: {call_name}")

        # Defense in depth: regex scan over raw code
        for danger in self.DANGEROUS_CALLS:
            if re.search(r"\b" + re.escape(danger) + r"\b", self.code):
                msg = f"Dangerous function invocation detected: {danger}"
                if msg not in violations:
                    violations.append(msg)

        return {
            "safe": len(violations) == 0,
            "violations": violations
        }

    def detect_computational_patterns(self) -> Dict[str, Any]:
        """Analyzes loops, vectorization, visualization, and data persistence patterns."""
        if not self.tree:
            return {
                "loop_count": 0,
                "has_plotting": False,
                "has_saving": False,
                "has_file_io": False,
                "scientific_libs": []
            }

        loop_count = 0
        has_plotting = False
        has_saving = False
        has_file_io = False

        for node in ast.walk(self.tree):
            if isinstance(node, (ast.For, ast.While)):
                loop_count += 1
            elif isinstance(node, ast.Call):
                name_str = ""
                if isinstance(node.func, ast.Name):
                    name_str = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name_str = node.func.attr

                if name_str in {"plot", "scatter", "subplots", "figure", "hist", "imshow"}:
                    has_plotting = True
                elif name_str in {"savefig", "savez", "to_csv", "dump"}:
                    has_saving = True
                elif name_str in {"open", "read", "write"}:
                    has_file_io = True

        imports = self.extract_imports()
        detected_sci_libs = [lib for lib in self.SCIENTIFIC_LIBRARIES if lib in imports["modules"] or lib in imports["aliases"]]

        return {
            "loop_count": loop_count,
            "has_plotting": has_plotting,
            "has_saving": has_saving,
            "has_file_io": has_file_io,
            "scientific_libs": detected_sci_libs
        }

    BUILTIN_NAMES: Set[str] = set(dir(builtins)) | {
        "__name__", "__file__", "__doc__", "__package__", "__spec__",
        "True", "False", "None", "self", "cls", "get_ipython"
    }

    def find_undefined_symbols(self) -> Set[str]:
        """
        Detects variables, modules, or functions that are referenced but never
        imported, assigned, or defined anywhere in the script.
        """
        if not self.tree:
            return set()

        defined_names: Set[str] = set(self.BUILTIN_NAMES)
        loaded_names: Set[str] = set()

        # Collect all defined symbols (imports, defs, classes, assigns, args, loop targets)
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    defined_names.add(alias.asname or alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    defined_names.add(alias.asname or alias.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined_names.add(node.name)
                if hasattr(node, "args"):
                    for a in getattr(node.args, "posonlyargs", []) + node.args.args + node.args.kwonlyargs:
                        defined_names.add(a.arg)
                    if node.args.vararg:
                        defined_names.add(node.args.vararg.arg)
                    if node.args.kwarg:
                        defined_names.add(node.args.kwarg.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined_names.add(node.id)
            elif isinstance(node, (ast.For, ast.comprehension)):
                if isinstance(node.target, ast.Name):
                    defined_names.add(node.target.id)
                elif isinstance(node.target, (ast.Tuple, ast.List)):
                    for elt in node.target.elts:
                        if isinstance(elt, ast.Name):
                            defined_names.add(elt.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined_names.add(node.name)
            elif isinstance(node, ast.withitem) and node.optional_vars:
                if isinstance(node.optional_vars, ast.Name):
                    defined_names.add(node.optional_vars.id)

        # Collect all loaded names
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                loaded_names.add(node.id)

        undefined = loaded_names - defined_names
        # Filter private or internal conventions
        return {u for u in undefined if not u.startswith("_")}

    def analyze(self) -> Dict[str, Any]:
        """Provides a comprehensive AST inspection report."""
        imports = self.extract_imports()
        functions = self.extract_functions()
        safety = self.check_safety()
        patterns = self.detect_computational_patterns()
        undefined_symbols = self.find_undefined_symbols()

        return {
            "syntax_valid": self.syntax_valid,
            "syntax_error": self.syntax_error_msg,
            "syntax_lineno": self.syntax_error_lineno,
            "imports": imports,
            "functions": functions,
            "safety": safety,
            "patterns": patterns,
            "undefined_symbols": sorted(list(undefined_symbols))
        }
