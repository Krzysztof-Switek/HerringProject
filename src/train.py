from engine.trainer import Trainer
from pathlib import Path

if __name__ == "__main__":
    try:
        project_root = Path(__file__).parent.parent
        trainer = Trainer(project_root=project_root)
        trainer.train()
    except Exception as e:
        print(f"Error: {e}")
        raise
