import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Katalogi do pominięcia
EXCLUDE_DIRS = {"__pycache__", ".venv", "checkpoints", "results", "tests"}
# Linie, które są dozwolone mimo obecności "model"
ALLOWLIST_PATTERNS = [
    r"\bbase_model\b",
    r"\bexpert_model\b",
    r"\bmodels\.model\b",
    r"from models import model",
    r"getattr\(models, cfg\.base_model",
    r"cfg\.base_model",
    r"self\.base_model",
]

def is_allowed(line: str) -> bool:
    return any(re.search(pattern, line) for pattern in ALLOWLIST_PATTERNS)

def test_no_ambiguous_model_usage():
    violations = []

    for root, dirs, files in os.walk(SRC_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for filename in files:
            if filename.endswith(".py"):
                file_path = Path(root) / filename
                with open(file_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, start=1):
                        if "model" in line and not is_allowed(line):
                            if not re.search(r"\b(base_model|expert_model)\b", line):
                                violations.append(f"{file_path.relative_to(PROJECT_ROOT)}:{i}: {line.strip()}")

    assert not violations, "❗ Ambiguous 'model' usages found:\n" + "\n".join(violations)
