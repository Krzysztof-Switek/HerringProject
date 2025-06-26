from pathlib import Path
from omegaconf import OmegaConf
from utils.path_manager import PathManager
from utils.Training_Prediction_Report import TrainingPredictionReport

def main():
    #####           USER INPUT          #####
    # Nazwa katalogu logów (np. "resnet50_standard_ce_multi_2025-06-26_11-16")
    log_dir_name = "resnet50_standard_ce_multi_2025-06-26_11-16"

    # Wyznacz katalog główny i załaduj config
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "src" / "config" / "config.yaml"
    cfg = OmegaConf.load(config_path)
    path_manager = PathManager(project_root, cfg)

    log_dir = path_manager.logs_dir() / log_dir_name
    metadata_path = path_manager.metadata_file()

    print(f"Używam log_dir: {log_dir}")
    print(f"Plik metadata: {metadata_path}")

    # Wyszukaj wymagane pliki
    predictions_path = next(log_dir.glob("*_predictions.xlsx"), None)
    metrics_path = next(log_dir.glob("*_training_metrics.csv"), None)
    augmentation_path = next(log_dir.glob("augmentation_summary_*.csv"), None)

    # Walidacja obecności plików
    if not predictions_path:
        raise FileNotFoundError(f"Nie znaleziono pliku *_predictions.xlsx w {log_dir}")
    if not metrics_path:
        print(f"⚠️ Nie znaleziono pliku *_training_metrics.csv w {log_dir} — raport będzie ograniczony do predykcji.")
    if not augmentation_path:
        print(f"⚠️ Nie znaleziono pliku augmentation_summary_*.csv w {log_dir} — nie będzie podsumowania augmentacji.")

    # Generowanie raportu
    report = TrainingPredictionReport(
        log_dir=log_dir,
        config_path=config_path,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        augmentation_path=augmentation_path
    )
    report.run()

if __name__ == "__main__":
    main()
