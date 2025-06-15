import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Uwzględniamy tylko te dwa pliki:
INCLUDED_FILES = {
    SRC_DIR / "models" / "model.py",
    SRC_DIR / "engine" / "predict_after_training.py",
    SRC_DIR / "engine" / "trainer_setup.py",
    SRC_DIR / "engine" / "trainer_metadata.py",
}


def is_allowed(line: str) -> bool:
    """
    Sprawdza, czy linia jest dopuszczalna mimo użycia 'model'
    """
    allowed_patterns = [
        r"\base_model\b",
        r"\expert_model\b",
        r"models\.",  # torchvision.models itp.
        r"model_to_column_map",
        r"model_name",
        r"model_path",
        r"model.eval\(",
        r"model.train\(",
        r"self\.model\b",  # dozwolone w klasach, można doprecyzować
    ]
    return any(re.search(pattern, line) for pattern in allowed_patterns)


def test_no_ambiguous_model_usage_in_core_files():
    violations = []

    for file_path in INCLUDED_FILES:
        if not file_path.exists():
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if "model" in line and not is_allowed(line):
                    if not re.search(r"\b(base_model|expert_model)\b", line):
                        relative_path = file_path.relative_to(PROJECT_ROOT)
                        violations.append(f"{relative_path}:{i}: {line.strip()}")

    assert not violations, (
        "❗ Ambiguous 'model' usages found in critical files:\n" + "\n".join(violations)
    )
