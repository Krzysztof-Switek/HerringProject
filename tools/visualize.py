"""
Standalone pipeline wizualizacji — GradCAM / GradCAM++ / Guided Backprop.

Uruchamiaj po treningu, gdy masz gotowy checkpoint w config.prediction.model_path.

Przykłady:
  # Domyślnie: test set z configa, wszystkie metody z configa
  .venv/Scripts/python tools/visualize.py

  # Limit do 10 obrazów (np. szybki test)
  .venv/Scripts/python tools/visualize.py --max-images 10

  # Własny katalog z obrazami
  .venv/Scripts/python tools/visualize.py --image-dir data/embedded/val/1

  # Tylko GradCAM, bez raportu HTML
  .venv/Scripts/python tools/visualize.py --methods gradcam --no-html

  # Własny checkpoint
  .venv/Scripts/python tools/visualize.py --model-path checkpoints/my_run/best.pth
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from omegaconf import OmegaConf

from src.engine.visualization_runner import VisualizationRunner
from src.utils.config_helpers import get_active_model_name
from src.utils.path_manager import PathManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generuje heatmapy GradCAM dla wytrenowanego modelu.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image-dir", type=Path, default=None,
        help="Katalog z obrazami (domyślnie: test set z config.data.root_dir/test)",
    )
    parser.add_argument(
        "--model-path", type=Path, default=None,
        help="Ścieżka do checkpointu (domyślnie: config.prediction.model_path)",
    )
    parser.add_argument(
        "--methods", nargs="+",
        choices=["gradcam", "gradcam++", "guided_backprop"],
        default=None,
        help="Metody wizualizacji (domyślnie: z config.visualization.methods)",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Limit liczby obrazów (domyślnie: wszystkie)",
    )
    parser.add_argument(
        "--no-html", action="store_true",
        help="Pomiń generowanie raportu HTML",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = OmegaConf.load(_PROJECT_ROOT / "src" / "config" / "config.yaml")

    # Nadpisz metody jeśli podano z CLI
    if args.methods:
        cfg.visualization.methods = args.methods

    path_manager = PathManager(_PROJECT_ROOT, cfg)

    # Katalog wejściowy — domyślnie test set
    if args.image_dir:
        image_dir = args.image_dir
    else:
        image_dir = path_manager.data_root() / cfg.data.test
    image_dir = Path(image_dir)

    if not image_dir.exists():
        print(f"BŁĄD: Katalog z obrazami nie istnieje: {image_dir}")
        sys.exit(1)

    # Katalog wyjściowy
    model_name = get_active_model_name(cfg)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"{model_name}_{timestamp}"
    output_dir = path_manager.gradcam_dir(run_name)

    print("=" * 60)
    print(f"Wizualizacja GradCAM")
    print(f"  Model:      {model_name}")
    print(f"  Obrazy:     {image_dir}")
    print(f"  Wyjście:    {output_dir}")
    print(f"  Metody:     {list(cfg.visualization.methods)}")
    print(f"  Limit:      {args.max_images or 'wszystkie'}")
    print("=" * 60)

    runner = VisualizationRunner(cfg, _PROJECT_ROOT, model_path=args.model_path)
    results = runner.run(
        image_dir=image_dir,
        output_dir=output_dir,
        max_images=args.max_images,
        generate_html=not args.no_html,
    )

    if results:
        scored = [r for r in results if r["correct"] is not None]
        if scored:
            acc = sum(1 for r in scored if r["correct"]) / len(scored) * 100
            print(f"\nDokładność: {acc:.1f}% ({sum(1 for r in scored if r['correct'])}/{len(scored)})")
        print(f"\nWyniki zapisano w: {output_dir}")
        if not args.no_html:
            print(f"Raport HTML:       {output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
