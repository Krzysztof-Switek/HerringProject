import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from engine.trainer_old_delete import Trainer

if __name__ == "__main__":
    try:
        project_root = Path(__file__).parent.parent
        trainer = Trainer(project_root=project_root)
        trainer.train()
    except Exception as e:
        print(f"Error: {e}")
        raise
