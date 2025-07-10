import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import argparse # Dodano argparse
from engine.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uruchamia trening modelu.")
    parser.add_argument(
        "--config",
        type=str,
        default=None, # Domyślnie None, Trainer użyje ścieżki z PathManager
        help="Ścieżka do pliku konfiguracyjnego YAML."
    )
    parser.add_argument(
        "--debug",
        action="store_true", # Ustawia na True, jeśli flaga jest obecna
        help="Uruchamia trening w trybie debug (np. mniej epok, szybsze zakończenie)."
    )
    args = parser.parse_args()

    try:
        project_root = Path(__file__).parent.parent
        # Przekaż argumenty do Trainer
        trainer = Trainer(
            project_root=project_root,
            config_path_override=args.config, # Przekaż ścieżkę z argumentu
            debug_mode=args.debug # Przekaż flagę debug
        )
        trainer.train()
    except Exception as e:
        print(f"Error: {e}")
        raise
