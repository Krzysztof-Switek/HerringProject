from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

project_root_str = str(PROJECT_ROOT)
src_root_str = str(SRC_ROOT)

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

if src_root_str not in sys.path:
    sys.path.insert(0, src_root_str)
