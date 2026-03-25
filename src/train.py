import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.engine.trainer import Trainer

if __name__ == "__main__":
    try:
        trainer = Trainer(project_root=_PROJECT_ROOT)
        trainer.train()
    except Exception as e:
        print(f"Error: {e}")
        raise
