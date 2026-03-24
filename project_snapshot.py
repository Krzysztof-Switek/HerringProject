from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# =============================================================================
# PROJECT SNAPSHOT TOOL — HERRINGPROJECT
# =============================================================================
#
# CEL
# - Zbudować realny snapshot aktualnego projektu bez zgadywania.
# - Ułatwić AI developerowi precyzyjną analizę kodu, pipeline'u i testów.
# - Wskazać kandydatów na:
#   * nieużywane pliki,
#   * nieużywane funkcje,
#   * słabe miejsca architektury,
#   * problemy logiczne wykrywalne heurystycznie.
#
# ZASADY
# 1. Never guess code structure.
# 2. Always rely on actual files.
# 3. One change at a time.
# 4. Snapshot ma być punktem startowym przed refaktorem/debugiem/analizą.
#
# DLA HERRINGPROJECT
# - priorytet: Python + testy + pipeline ML/biologiczny
# - ważne obszary:
#   * data loading
#   * losses / criteria
#   * pipeline
#   * model
#   * population integrity
#   * logger
#
# =============================================================================


# =========================
# ⚙️ KONFIGURACJA
# =========================
PROJECT_ROOT = Path(".").resolve()

OUTPUT_TXT = "PROJECT_MAP.txt"
OUTPUT_JSON = "project_snapshot.json"
OUTPUT_DOT = "import_graph.dot"

SOURCE_DIR_CANDIDATES = ["src", "."]
TEST_DIR_NAMES = {"tests", "test"}

INCLUDE_EXTENSIONS = {
    ".py": "Python",
    ".json": "JSON",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".toml": "TOML",
    ".ini": "INI",
    ".cfg": "CFG",
    ".txt": "TEXT",
    ".md": "Markdown",
}

EXCLUDE_DIRS = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "coverage",
    "site-packages",
    ".eggs",
}

EXCLUDE_FILE_NAMES = {
    OUTPUT_TXT,
    OUTPUT_JSON,
    OUTPUT_DOT,
}

IGNORE_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff",
    ".npy", ".npz", ".pkl", ".pickle", ".joblib", ".pt", ".pth",
    ".zip", ".tar", ".gz", ".7z", ".pdf", ".xlsx", ".xls", ".parquet",
}

PIPELINE_KEYWORDS = [
    "dataset",
    "dataloader",
    "loader",
    "pipeline",
    "train",
    "training",
    "evaluate",
    "eval",
    "loss",
    "criterion",
    "logger",
    "model",
    "population",
    "integrity",
    "otolith",
    "herring",
]

SUSPICIOUS_PATTERNS = [
    "TODO",
    "FIXME",
    "HACK",
    "XXX",
]

COMMON_STD_LIBS = {
    "os", "sys", "math", "json", "pathlib", "typing", "collections",
    "itertools", "functools", "statistics", "re", "subprocess", "logging",
    "datetime", "time", "random", "hashlib", "csv", "copy", "glob",
    "traceback", "shutil", "tempfile", "argparse", "dataclasses",
    "unittest", "ast", "inspect", "enum", "threading", "multiprocessing",
}

# mapowanie pakietów -> import root
REQ_IMPORT_ALIASES = {
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
    "opencv-contrib-python": "cv2",
    "pyyaml": "yaml",
    "pillow": "PIL",
    "tensorflow-cpu": "tensorflow",
    "python-dotenv": "dotenv",
}


# =========================
# 🧱 MODELE DANYCH
# =========================
@dataclass
class FunctionInfo:
    name: str
    lineno: int
    end_lineno: Optional[int]
    decorators: List[str]
    args: List[str]
    is_async: bool
    docstring: bool
    calls: List[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    name: str
    lineno: int
    end_lineno: Optional[int]
    bases: List[str]
    methods: List[FunctionInfo] = field(default_factory=list)
    docstring: bool = False


@dataclass
class PythonFileInfo:
    path: str
    module_name: str
    role: str
    size_bytes: int
    lines: int
    imports: List[str]
    imported_symbols: List[str]
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    top_level_calls: List[str]
    top_level_assignments: List[str]
    has_main_guard: bool
    has_docstring: bool
    suspicious: List[Dict[str, Any]]
    tests_defined: List[str]
    pytest_fixtures: List[str]
    strings_of_interest: List[str]
    syntax_error: Optional[str] = None


@dataclass
class GenericFileInfo:
    path: str
    role: str
    type: str
    size_bytes: int
    lines: int


# =========================
# 🧰 NARZĘDZIA
# =========================
def safe_read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
    except Exception:
        return None


def count_lines(text: str) -> int:
    return text.count("\n") + 1 if text else 0


def is_test_path(path: Path) -> bool:
    parts = set(path.parts)
    if parts & TEST_DIR_NAMES:
        return True
    return path.name.startswith("test_") or path.name.endswith("_test.py")


def is_python_source_path(path: Path) -> bool:
    return path.suffix == ".py" and not is_test_path(path)


def classify_role(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT).as_posix()

    if is_test_path(path):
        return "TEST"
    if rel.startswith("src/"):
        return "SOURCE"
    if rel.startswith("data/"):
        return "DATA"
    if rel.startswith("delete/"):
        return "AUXILIARY"
    return "PROJECT"


def should_skip_dir(dirname: str) -> bool:
    return dirname in EXCLUDE_DIRS


def should_include_file(path: Path) -> bool:
    if path.name in EXCLUDE_FILE_NAMES:
        return False
    if path.suffix.lower() in IGNORE_BINARY_EXTENSIONS:
        return False
    if path.suffix.lower() in INCLUDE_EXTENSIONS:
        return True
    if path.name == "requirements.txt":
        return True
    return False


def iter_project_files(root: Path) -> List[Path]:
    collected = []
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]
        for fname in sorted(files):
            path = Path(current_root) / fname
            if should_include_file(path):
                collected.append(path.resolve())
    return sorted(collected)


def detect_source_root(files: List[Path]) -> Path:
    src = PROJECT_ROOT / "src"
    if src.exists() and src.is_dir():
        return src.resolve()
    return PROJECT_ROOT


def path_to_module_name(path: Path, source_root: Path) -> str:
    rel_project = path.relative_to(PROJECT_ROOT)

    if rel_project.as_posix().startswith("src/"):
        rel = path.relative_to(source_root)
        module = rel.with_suffix("").as_posix().replace("/", ".")
        return module

    if is_test_path(path):
        return rel_project.with_suffix("").as_posix().replace("/", ".")

    return rel_project.with_suffix("").as_posix().replace("/", ".")


def normalize_import_root(import_name: str) -> str:
    return import_name.split(".")[0].strip()


def looks_like_internal_import(import_name: str, known_modules: Set[str], module_roots: Set[str]) -> bool:
    if import_name in known_modules:
        return True
    root = normalize_import_root(import_name)
    if root in module_roots:
        return True
    return False


def safe_run_git(args: List[str]) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            shell=False,
        )
        if proc.returncode == 0:
            return True, proc.stdout.strip()
        return False, (proc.stderr or proc.stdout).strip()
    except Exception as e:
        return False, str(e)


# =========================
# 🧠 AST ANALIZA PYTHONA
# =========================
class PythonAstAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: List[str] = []
        self.imported_symbols: List[str] = []
        self.functions: List[FunctionInfo] = []
        self.classes: List[ClassInfo] = []
        self.top_level_calls: List[str] = []
        self.top_level_assignments: List[str] = []
        self.tests_defined: List[str] = []
        self.pytest_fixtures: List[str] = []

        self._class_stack: List[ClassInfo] = []
        self._function_stack: List[FunctionInfo] = []
        self._module_docstring = False

    @property
    def has_docstring(self) -> bool:
        return self._module_docstring

    def visit_Module(self, node: ast.Module) -> Any:
        self._module_docstring = ast.get_docstring(node) is not None
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            self.imports.append(alias.name)
            self.imported_symbols.append(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        module = node.module or ""
        if node.level and module:
            import_repr = "." * node.level + module
        elif node.level:
            import_repr = "." * node.level
        else:
            import_repr = module

        if import_repr:
            self.imports.append(import_repr)

        for alias in node.names:
            symbol = f"{module}.{alias.name}" if module else alias.name
            self.imported_symbols.append(alias.asname or symbol)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        if not self._function_stack and not self._class_stack:
            for target in node.targets:
                name = self._get_target_name(target)
                if name:
                    self.top_level_assignments.append(name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if not self._function_stack and not self._class_stack:
            name = self._get_target_name(node.target)
            if name:
                self.top_level_assignments.append(name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        name = self._call_name(node)
        if name:
            if self._function_stack:
                self._function_stack[-1].calls.append(name)
            elif not self._class_stack:
                self.top_level_calls.append(name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        info = self._build_function_info(node, is_async=False)
        self._register_function(info, node)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        info = self._build_function_info(node, is_async=True)
        self._register_function(info, node)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        info = ClassInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", None),
            bases=[self._expr_to_str(base) for base in node.bases],
            methods=[],
            docstring=ast.get_docstring(node) is not None,
        )
        self.classes.append(info)
        self._class_stack.append(info)
        self.generic_visit(node)
        self._class_stack.pop()

    def _register_function(self, info: FunctionInfo, node: ast.AST) -> None:
        if self._class_stack:
            self._class_stack[-1].methods.append(info)
        else:
            self.functions.append(info)
            if info.name.startswith("test_"):
                self.tests_defined.append(info.name)

        if any(dec.endswith("fixture") or dec == "pytest.fixture" for dec in info.decorators):
            self.pytest_fixtures.append(info.name)

        self._function_stack.append(info)

    def _build_function_info(self, node: ast.AST, is_async: bool) -> FunctionInfo:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        decorators = [self._expr_to_str(d) for d in node.decorator_list]
        args = [a.arg for a in node.args.args]
        return FunctionInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", None),
            decorators=decorators,
            args=args,
            is_async=is_async,
            docstring=ast.get_docstring(node) is not None,
            calls=[],
        )

    def _get_target_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return self._expr_to_str(node)
        return None

    def _call_name(self, node: ast.Call) -> Optional[str]:
        return self._expr_to_str(node.func)

    def _expr_to_str(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._expr_to_str(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        if isinstance(node, ast.Call):
            return self._expr_to_str(node.func)
        if isinstance(node, ast.Subscript):
            return self._expr_to_str(node.value)
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Lambda):
            return "<lambda>"
        if isinstance(node, ast.Tuple):
            return "(" + ", ".join(self._expr_to_str(e) for e in node.elts) + ")"
        if isinstance(node, ast.List):
            return "[" + ", ".join(self._expr_to_str(e) for e in node.elts) + "]"
        try:
            return ast.unparse(node)
        except Exception:
            return type(node).__name__


# =========================
# 🔎 ANALIZA PLIKÓW
# =========================
def find_suspicious_patterns(path: Path, text: str, role: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    lines = text.splitlines()

    for idx, line in enumerate(lines, 1):
        stripped = line.strip()

        for token in SUSPICIOUS_PATTERNS:
            if token in line:
                findings.append({
                    "type": "marker",
                    "line": idx,
                    "value": token,
                    "context": stripped[:180],
                })

        if re.match(r"^\s*except\s*:\s*$", line):
            findings.append({
                "type": "bare_except",
                "line": idx,
                "value": "except:",
                "context": stripped,
            })

        if re.match(r"^\s*pass\s*$", line):
            findings.append({
                "type": "pass",
                "line": idx,
                "value": "pass",
                "context": stripped,
            })

        if role != "TEST" and re.search(r"\bprint\s*\(", line):
            findings.append({
                "type": "print_in_non_test",
                "line": idx,
                "value": "print(",
                "context": stripped[:180],
            })

        if re.search(r"\bpdb\.set_trace\s*\(", line):
            findings.append({
                "type": "debug_breakpoint",
                "line": idx,
                "value": "pdb.set_trace",
                "context": stripped[:180],
            })

    return findings


def extract_strings_of_interest(text: str) -> List[str]:
    hits: Set[str] = set()
    for keyword in PIPELINE_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE):
            hits.add(keyword)
    return sorted(hits)


def analyze_python_file(path: Path, source_root: Path) -> PythonFileInfo:
    text = safe_read_text(path) or ""
    role = classify_role(path)
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    module_name = path_to_module_name(path, source_root)

    suspicious = find_suspicious_patterns(path, text, role)
    strings_of_interest = extract_strings_of_interest(text)

    try:
        tree = ast.parse(text)
        analyzer = PythonAstAnalyzer()
        analyzer.visit(tree)

        return PythonFileInfo(
            path=rel,
            module_name=module_name,
            role=role,
            size_bytes=path.stat().st_size,
            lines=count_lines(text),
            imports=sorted(set(analyzer.imports)),
            imported_symbols=sorted(set(analyzer.imported_symbols)),
            functions=analyzer.functions,
            classes=analyzer.classes,
            top_level_calls=sorted(set(analyzer.top_level_calls)),
            top_level_assignments=sorted(set(analyzer.top_level_assignments)),
            has_main_guard='if __name__ == "__main__":' in text or "if __name__ == '__main__':" in text,
            has_docstring=analyzer.has_docstring,
            suspicious=suspicious,
            tests_defined=sorted(set(analyzer.tests_defined)),
            pytest_fixtures=sorted(set(analyzer.pytest_fixtures)),
            strings_of_interest=strings_of_interest,
            syntax_error=None,
        )
    except SyntaxError as e:
        return PythonFileInfo(
            path=rel,
            module_name=module_name,
            role=role,
            size_bytes=path.stat().st_size,
            lines=count_lines(text),
            imports=[],
            imported_symbols=[],
            functions=[],
            classes=[],
            top_level_calls=[],
            top_level_assignments=[],
            has_main_guard=False,
            has_docstring=False,
            suspicious=suspicious,
            tests_defined=[],
            pytest_fixtures=[],
            strings_of_interest=strings_of_interest,
            syntax_error=f"{e.msg} (line {e.lineno})",
        )


def analyze_generic_file(path: Path) -> GenericFileInfo:
    text = safe_read_text(path)
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    file_type = INCLUDE_EXTENSIONS.get(path.suffix.lower(), "Generic")

    return GenericFileInfo(
        path=rel,
        role=classify_role(path),
        type=file_type,
        size_bytes=path.stat().st_size,
        lines=count_lines(text or ""),
    )


# =========================
# 📦 REQUIREMENTS / IMPORTY
# =========================
def parse_requirements_txt(path: Path) -> List[str]:
    if not path.exists():
        return []

    text = safe_read_text(path)
    if not text:
        return []

    packages = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        s = re.split(r"[;<>=\[]", s, maxsplit=1)[0].strip()
        if s:
            packages.append(s)
    return packages


def normalize_requirement_to_import_root(req_name: str) -> str:
    lower = req_name.lower()
    if lower in REQ_IMPORT_ALIASES:
        return REQ_IMPORT_ALIASES[lower]
    return lower.replace("-", "_")


# =========================
# 🔗 GRAF ZALEŻNOŚCI
# =========================
def build_import_graph(
    python_infos: List[PythonFileInfo],
) -> Dict[str, Set[str]]:
    known_modules = {info.module_name for info in python_infos}
    module_roots = {normalize_import_root(m) for m in known_modules}

    graph: Dict[str, Set[str]] = defaultdict(set)

    for info in python_infos:
        for imp in info.imports:
            imp_clean = imp.lstrip(".")
            if not imp_clean:
                continue

            if looks_like_internal_import(imp_clean, known_modules, module_roots):
                root = normalize_import_root(imp_clean)

                # Prefer exact module if present, otherwise root package/module
                if imp_clean in known_modules:
                    graph[info.module_name].add(imp_clean)
                else:
                    matches = [m for m in known_modules if m == root or m.startswith(root + ".")]
                    if matches:
                        graph[info.module_name].add(sorted(matches)[0])

    for info in python_infos:
        graph.setdefault(info.module_name, set())

    return graph


def reverse_graph(graph: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    rev: Dict[str, Set[str]] = defaultdict(set)
    for src, targets in graph.items():
        rev.setdefault(src, set())
        for dst in targets:
            rev[dst].add(src)
    return rev


# =========================
# 🧪 TESTY
# =========================
def build_test_to_source_links(
    python_infos: List[PythonFileInfo],
) -> Dict[str, List[str]]:
    source_modules = {info.module_name for info in python_infos if info.role != "TEST"}
    links: Dict[str, List[str]] = {}

    for info in python_infos:
        if info.role != "TEST":
            continue

        candidates: Set[str] = set()

        # import-based
        for imp in info.imports:
            imp_clean = imp.lstrip(".")
            if imp_clean in source_modules:
                candidates.add(imp_clean)
            else:
                root = normalize_import_root(imp_clean)
                for mod in source_modules:
                    if mod == root or mod.startswith(root + "."):
                        candidates.add(mod)

        # filename-based heuristic: dataset_test.py -> dataset
        stem = Path(info.path).stem
        stem = re.sub(r"(^test_|_test$)", "", stem)
        for mod in source_modules:
            mod_leaf = mod.split(".")[-1]
            if mod_leaf == stem or stem in mod_leaf or mod_leaf in stem:
                candidates.add(mod)

        links[info.path] = sorted(candidates)

    return links


# =========================
# 🪤 NIEUŻYWANE PLIKI / FUNKCJE
# =========================
def find_orphan_python_modules(
    python_infos: List[PythonFileInfo],
    graph: Dict[str, Set[str]],
) -> List[Dict[str, Any]]:
    rev = reverse_graph(graph)

    results = []
    for info in python_infos:
        if info.role == "TEST":
            continue

        inbound = rev.get(info.module_name, set())
        reasons = []

        if not inbound:
            reasons.append("no internal imports pointing to module")
        if not info.has_main_guard and not info.top_level_calls:
            reasons.append("no explicit runtime entrypoint")
        if "init" in Path(info.path).name.lower():
            continue

        if reasons:
            results.append({
                "module": info.module_name,
                "path": info.path,
                "reasons": reasons,
            })

    return sorted(results, key=lambda x: x["path"])


def find_potentially_unused_functions(
    python_infos: List[PythonFileInfo],
) -> List[Dict[str, Any]]:
    called_names = Counter()

    for info in python_infos:
        for fn in info.functions:
            for call in fn.calls:
                called_names[call.split(".")[-1]] += 1
        for cls in info.classes:
            for method in cls.methods:
                for call in method.calls:
                    called_names[call.split(".")[-1]] += 1
        for call in info.top_level_calls:
            called_names[call.split(".")[-1]] += 1

    results = []

    for info in python_infos:
        if info.role == "TEST":
            continue

        for fn in info.functions:
            if fn.name.startswith("_"):
                continue
            if fn.name in {"main"}:
                continue
            if called_names[fn.name] == 0:
                results.append({
                    "path": info.path,
                    "function": fn.name,
                    "line": fn.lineno,
                    "reason": "no call name detected in project",
                })

    return sorted(results, key=lambda x: (x["path"], x["line"]))


# =========================
# ⚠️ HEURYSTYKI LOGICZNE / ARCHITEKTURA
# =========================
def detect_ml_hotspots(python_infos: List[PythonFileInfo]) -> Dict[str, List[str]]:
    hotspots = defaultdict(list)

    for info in python_infos:
        joined = " ".join(info.strings_of_interest).lower()
        path_lower = info.path.lower()

        if "dataset" in joined or "dataset" in path_lower:
            hotspots["dataset"].append(info.path)
        if "pipeline" in joined or "pipeline" in path_lower:
            hotspots["pipeline"].append(info.path)
        if "loss" in joined or "loss" in path_lower or "criterion" in joined:
            hotspots["losses"].append(info.path)
        if "model" in joined or "model" in path_lower:
            hotspots["model"].append(info.path)
        if "logger" in joined or "logger" in path_lower:
            hotspots["logger"].append(info.path)
        if "population" in joined or "population" in path_lower:
            hotspots["population"].append(info.path)
        if "integrity" in joined or "integrity" in path_lower:
            hotspots["integrity"].append(info.path)

    for key in list(hotspots.keys()):
        hotspots[key] = sorted(set(hotspots[key]))

    return dict(hotspots)


def detect_high_risk_files(python_infos: List[PythonFileInfo]) -> List[Dict[str, Any]]:
    results = []

    for info in python_infos:
        score = 0
        reasons = []

        if info.syntax_error:
            score += 100
            reasons.append(f"syntax_error: {info.syntax_error}")

        if len(info.suspicious) >= 3:
            score += 20
            reasons.append(f"{len(info.suspicious)} suspicious markers")

        if info.role != "TEST" and not info.has_docstring:
            score += 5
            reasons.append("missing module docstring")

        if info.lines > 400:
            score += 15
            reasons.append("large file > 400 LOC")

        total_functions = len(info.functions) + sum(len(c.methods) for c in info.classes)
        if total_functions > 20:
            score += 10
            reasons.append("many functions/methods")

        if info.has_main_guard:
            score += 5
            reasons.append("entrypoint / executable module")

        if any(s["type"] == "bare_except" for s in info.suspicious):
            score += 15
            reasons.append("bare except")

        if score > 0:
            results.append({
                "path": info.path,
                "score": score,
                "reasons": reasons,
            })

    return sorted(results, key=lambda x: (-x["score"], x["path"]))


def detect_files_without_tests(
    python_infos: List[PythonFileInfo],
    test_links: Dict[str, List[str]],
) -> List[str]:
    covered: Set[str] = set()
    for linked in test_links.values():
        covered.update(linked)

    uncovered = []
    for info in python_infos:
        if info.role == "TEST":
            continue
        if info.module_name not in covered:
            uncovered.append(info.path)

    return sorted(uncovered)


# =========================
# 🧾 RAPORTY
# =========================
def generate_tree(files: List[Path]) -> str:
    tree: Dict[str, List[str]] = defaultdict(list)
    for path in files:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        parent = str(Path(rel).parent).replace("\\", "/")
        tree[parent].append(Path(rel).name)

    lines = []
    for directory in sorted(tree.keys()):
        title = "." if directory == "." else directory
        lines.append(f"[DIR] {title}")
        for fname in sorted(tree[directory]):
            lines.append(f"  - {fname}")
    return "\n".join(lines)


def render_git_summary() -> str:
    ok_branch, branch = safe_run_git(["branch", "--show-current"])
    ok_status, status = safe_run_git(["status", "--short"])
    ok_head, head = safe_run_git(["rev-parse", "--short", "HEAD"])
    ok_last, last = safe_run_git(["log", "-1", "--oneline"])

    out = []
    out.append("=== GIT SUMMARY ===")
    out.append(f"- branch: {branch if ok_branch else '(unavailable)'}")
    out.append(f"- HEAD: {head if ok_head else '(unavailable)'}")
    out.append(f"- last commit: {last if ok_last else '(unavailable)'}")
    out.append("- status:")
    if ok_status:
        if status:
            for line in status.splitlines():
                out.append(f"  * {line}")
        else:
            out.append("  * working tree clean")
    else:
        out.append(f"  * unavailable: {status}")
    return "\n".join(out)


def render_requirements_summary(
    requirements: List[str],
    python_infos: List[PythonFileInfo],
) -> str:
    used_import_roots = sorted({
        normalize_import_root(imp)
        for info in python_infos
        for imp in info.imports
        if normalize_import_root(imp) not in COMMON_STD_LIBS
    })

    req_to_import = {req: normalize_requirement_to_import_root(req) for req in requirements}

    missing_in_requirements = []
    declared_but_not_seen = []

    req_import_roots = set(req_to_import.values())

    for root in used_import_roots:
        if root not in req_import_roots:
            missing_in_requirements.append(root)

    for req, import_root in req_to_import.items():
        if import_root not in used_import_roots:
            declared_but_not_seen.append(req)

    out = []
    out.append("=== REQUIREMENTS SUMMARY ===")
    out.append(f"- requirements.txt packages: {len(requirements)}")
    if requirements:
        for req in requirements:
            out.append(f"  * {req}")

    out.append("- imported third-party roots detected:")
    if used_import_roots:
        for root in used_import_roots:
            out.append(f"  * {root}")
    else:
        out.append("  * none")

    out.append("- possible imports missing in requirements:")
    if missing_in_requirements:
        for item in missing_in_requirements:
            out.append(f"  * {item}")
    else:
        out.append("  * none")

    out.append("- requirements declared but not seen in imports:")
    if declared_but_not_seen:
        for item in declared_but_not_seen:
            out.append(f"  * {item}")
    else:
        out.append("  * none")

    return "\n".join(out)


def render_project_summary(
    files: List[Path],
    python_infos: List[PythonFileInfo],
    generic_infos: List[GenericFileInfo],
) -> str:
    role_counter = Counter()
    for info in python_infos:
        role_counter[info.role] += 1
    for info in generic_infos:
        role_counter[info.role] += 1

    out = []
    out.append("=== PROJECT SUMMARY ===")
    out.append(f"- root: {PROJECT_ROOT.as_posix()}")
    out.append(f"- all included files: {len(files)}")
    out.append(f"- python files: {len(python_infos)}")
    out.append(f"- non-python included files: {len(generic_infos)}")
    out.append("- roles:")
    for role, count in sorted(role_counter.items()):
        out.append(f"  * {role}: {count}")

    source_files = [i for i in python_infos if i.role == "SOURCE"]
    test_files = [i for i in python_infos if i.role == "TEST"]

    out.append(f"- source python files: {len(source_files)}")
    out.append(f"- test python files: {len(test_files)}")
    out.append(f"- main guards detected: {sum(1 for i in python_infos if i.has_main_guard)}")
    out.append(f"- syntax errors detected: {sum(1 for i in python_infos if i.syntax_error)}")
    return "\n".join(out)


def render_python_files_section(python_infos: List[PythonFileInfo]) -> str:
    out = []
    out.append("=== PYTHON FILE MAP ===")

    for info in sorted(python_infos, key=lambda x: x.path):
        out.append(f"\n[PY] {info.path}")
        out.append(f"  - module: {info.module_name}")
        out.append(f"  - role: {info.role}")
        out.append(f"  - lines: {info.lines}")
        out.append(f"  - size_bytes: {info.size_bytes}")
        out.append(f"  - module_docstring: {info.has_docstring}")
        out.append(f"  - main_guard: {info.has_main_guard}")

        if info.syntax_error:
            out.append(f"  - syntax_error: {info.syntax_error}")

        if info.imports:
            out.append("  - imports:")
            for imp in info.imports[:30]:
                out.append(f"      * {imp}")

        if info.top_level_assignments:
            out.append("  - top_level_assignments:")
            for item in info.top_level_assignments[:20]:
                out.append(f"      * {item}")

        if info.top_level_calls:
            out.append("  - top_level_calls:")
            for item in info.top_level_calls[:20]:
                out.append(f"      * {item}")

        if info.functions:
            out.append("  - functions:")
            for fn in info.functions[:40]:
                out.append(
                    f"      * {fn.name}(...) line={fn.lineno} async={fn.is_async} "
                    f"docstring={fn.docstring} decorators={fn.decorators}"
                )

        if info.classes:
            out.append("  - classes:")
            for cls in info.classes[:20]:
                out.append(
                    f"      * {cls.name} line={cls.lineno} bases={cls.bases} docstring={cls.docstring}"
                )
                for method in cls.methods[:20]:
                    out.append(
                        f"          - {method.name}(...) line={method.lineno} async={method.is_async} "
                        f"docstring={method.docstring}"
                    )

        if info.tests_defined:
            out.append("  - tests_defined:")
            for test_name in info.tests_defined:
                out.append(f"      * {test_name}")

        if info.pytest_fixtures:
            out.append("  - pytest_fixtures:")
            for fx in info.pytest_fixtures:
                out.append(f"      * {fx}")

        if info.strings_of_interest:
            out.append("  - ml_keywords:")
            for kw in info.strings_of_interest:
                out.append(f"      * {kw}")

        if info.suspicious:
            out.append("  - suspicious:")
            for s in info.suspicious[:40]:
                out.append(
                    f"      * {s['type']} @ line {s['line']} | {s['value']} | {s['context']}"
                )

    return "\n".join(out)


def render_generic_files_section(generic_infos: List[GenericFileInfo]) -> str:
    out = []
    out.append("=== NON-PYTHON FILE MAP ===")
    for info in sorted(generic_infos, key=lambda x: x.path):
        out.append(f"- {info.path} | type={info.type} | role={info.role} | lines={info.lines} | size={info.size_bytes}")
    return "\n".join(out)


def render_import_graph_section(graph: Dict[str, Set[str]]) -> str:
    out = []
    out.append("=== INTERNAL IMPORT GRAPH ===")
    if not graph:
        out.append("- none")
        return "\n".join(out)

    for src in sorted(graph.keys()):
        targets = sorted(graph[src])
        if not targets:
            out.append(f"- {src} -> []")
        else:
            out.append(f"- {src} ->")
            for dst in targets:
                out.append(f"    * {dst}")
    return "\n".join(out)


def render_test_links_section(test_links: Dict[str, List[str]]) -> str:
    out = []
    out.append("=== TEST -> SOURCE LINKS ===")
    if not test_links:
        out.append("- none")
        return "\n".join(out)

    for test_path, linked in sorted(test_links.items()):
        out.append(f"- {test_path}")
        if linked:
            for item in linked:
                out.append(f"    * {item}")
        else:
            out.append("    * (no linked source module detected)")
    return "\n".join(out)


def render_orphans_section(
    orphan_modules: List[Dict[str, Any]],
    unused_functions: List[Dict[str, Any]],
    files_without_tests: List[str],
) -> str:
    out = []
    out.append("=== ORPHANS / UNUSED CANDIDATES ===")

    out.append("- python modules potentially orphaned:")
    if orphan_modules:
        for item in orphan_modules:
            out.append(f"  * {item['path']} ({item['module']})")
            for r in item["reasons"]:
                out.append(f"      - {r}")
    else:
        out.append("  * none")

    out.append("- functions potentially unused:")
    if unused_functions:
        for item in unused_functions[:200]:
            out.append(f"  * {item['path']}:{item['line']} -> {item['function']} | {item['reason']}")
    else:
        out.append("  * none")

    out.append("- source files with no linked tests:")
    if files_without_tests:
        for item in files_without_tests:
            out.append(f"  * {item}")
    else:
        out.append("  * none")

    return "\n".join(out)


def render_hotspots_section(
    hotspots: Dict[str, List[str]],
    high_risk: List[Dict[str, Any]],
) -> str:
    out = []
    out.append("=== ML / PIPELINE HOTSPOTS ===")

    if hotspots:
        for name, files in sorted(hotspots.items()):
            out.append(f"- {name}:")
            for f in files:
                out.append(f"  * {f}")
    else:
        out.append("- none")

    out.append("\n=== HIGH RISK FILES ===")
    if high_risk:
        for item in high_risk[:50]:
            out.append(f"- {item['path']} | score={item['score']}")
            for reason in item["reasons"]:
                out.append(f"  * {reason}")
    else:
        out.append("- none")

    return "\n".join(out)


def render_recommended_review_order(
    hotspots: Dict[str, List[str]],
    high_risk: List[Dict[str, Any]],
    orphan_modules: List[Dict[str, Any]],
) -> str:
    out = []
    out.append("=== RECOMMENDED REVIEW ORDER FOR AI DEVELOPER ===")
    out.append("1. requirements.txt")
    out.append("2. pliki pipeline / dataset / losses / model")
    out.append("3. testy integracyjne i testy krytyczne")
    out.append("4. pliki o najwyższym risk score")
    out.append("5. kandydaci na orphan modules")
    out.append("6. pliki z bare except / pass / TODO / FIXME")

    interesting = []
    for key in ["pipeline", "dataset", "losses", "model", "population", "integrity", "logger"]:
        interesting.extend(hotspots.get(key, []))

    interesting = list(dict.fromkeys(interesting))

    if interesting:
        out.append("- suggested files first:")
        for f in interesting[:20]:
            out.append(f"  * {f}")

    if high_risk:
        out.append("- top risk files:")
        for item in high_risk[:10]:
            out.append(f"  * {item['path']}")

    if orphan_modules:
        out.append("- orphan candidates:")
        for item in orphan_modules[:10]:
            out.append(f"  * {item['path']}")

    return "\n".join(out)


def write_dot_graph(graph: Dict[str, Set[str]], output_path: Path) -> None:
    lines = ["digraph import_graph {", '  rankdir="LR";', '  node [shape=box];']
    for src in sorted(graph.keys()):
        if not graph[src]:
            lines.append(f'  "{src}";')
        for dst in sorted(graph[src]):
            lines.append(f'  "{src}" -> "{dst}";')
    lines.append("}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


# =========================
# 🏗️ BUDOWA SNAPSHOTA
# =========================
def build_snapshot() -> Dict[str, Any]:
    files = iter_project_files(PROJECT_ROOT)
    source_root = detect_source_root(files)

    python_infos: List[PythonFileInfo] = []
    generic_infos: List[GenericFileInfo] = []

    for path in files:
        if path.suffix.lower() == ".py":
            python_infos.append(analyze_python_file(path, source_root))
        else:
            generic_infos.append(analyze_generic_file(path))

    graph = build_import_graph(python_infos)
    test_links = build_test_to_source_links(python_infos)

    orphan_modules = find_orphan_python_modules(python_infos, graph)
    unused_functions = find_potentially_unused_functions(python_infos)
    files_without_tests = detect_files_without_tests(python_infos, test_links)
    hotspots = detect_ml_hotspots(python_infos)
    high_risk = detect_high_risk_files(python_infos)

    requirements_path = PROJECT_ROOT / "requirements.txt"
    requirements = parse_requirements_txt(requirements_path)

    snapshot = {
        "project_root": PROJECT_ROOT.as_posix(),
        "source_root": source_root.as_posix(),
        "files_total": len(files),
        "python_files_total": len(python_infos),
        "generic_files_total": len(generic_infos),
        "requirements": requirements,
        "python_files": [asdict(info) for info in python_infos],
        "generic_files": [asdict(info) for info in generic_infos],
        "import_graph": {k: sorted(v) for k, v in graph.items()},
        "test_links": test_links,
        "orphan_modules": orphan_modules,
        "unused_functions": unused_functions,
        "files_without_tests": files_without_tests,
        "hotspots": hotspots,
        "high_risk_files": high_risk,
    }
    return snapshot


def render_full_report(snapshot: Dict[str, Any], files: List[Path]) -> str:
    python_infos = [PythonFileInfo(**{
        **item,
        "functions": [FunctionInfo(**f) for f in item["functions"]],
        "classes": [
            ClassInfo(
                name=c["name"],
                lineno=c["lineno"],
                end_lineno=c["end_lineno"],
                bases=c["bases"],
                methods=[FunctionInfo(**m) for m in c["methods"]],
                docstring=c["docstring"],
            )
            for c in item["classes"]
        ],
    }) for item in snapshot["python_files"]]

    generic_infos = [GenericFileInfo(**item) for item in snapshot["generic_files"]]

    parts = [
        "=" * 100,
        "PROJECT SNAPSHOT TOOL — HERRINGPROJECT",
        "=" * 100,
        render_git_summary(),
        "",
        render_project_summary(files, python_infos, generic_infos),
        "",
        "=== FILE TREE ===",
        generate_tree(files),
        "",
        render_requirements_summary(snapshot["requirements"], python_infos),
        "",
        render_import_graph_section({
            k: set(v) for k, v in snapshot["import_graph"].items()
        }),
        "",
        render_test_links_section(snapshot["test_links"]),
        "",
        render_orphans_section(
            snapshot["orphan_modules"],
            snapshot["unused_functions"],
            snapshot["files_without_tests"],
        ),
        "",
        render_hotspots_section(snapshot["hotspots"], snapshot["high_risk_files"]),
        "",
        render_python_files_section(python_infos),
        "",
        render_generic_files_section(generic_infos),
        "",
        render_recommended_review_order(
            snapshot["hotspots"],
            snapshot["high_risk_files"],
            snapshot["orphan_modules"],
        ),
        "",
        "=" * 100,
        "END OF SNAPSHOT",
        "=" * 100,
    ]
    return "\n".join(parts)


# =========================
# ▶ MAIN
# =========================
def main() -> int:
    print("🔍 Building HerringProject snapshot...")

    files = iter_project_files(PROJECT_ROOT)
    snapshot = build_snapshot()

    report = render_full_report(snapshot, files)

    txt_path = PROJECT_ROOT / OUTPUT_TXT
    json_path = PROJECT_ROOT / OUTPUT_JSON
    dot_path = PROJECT_ROOT / OUTPUT_DOT

    txt_path.write_text(report, encoding="utf-8")
    json_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    write_dot_graph({k: set(v) for k, v in snapshot["import_graph"].items()}, dot_path)

    print(f"✅ Written: {txt_path.name}")
    print(f"✅ Written: {json_path.name}")
    print(f"✅ Written: {dot_path.name}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())