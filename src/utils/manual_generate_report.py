from pathlib import Path
from omegaconf import OmegaConf
from utils.path_manager import PathManager
from utils.Training_Prediction_Report import TrainingPredictionReport

def main():
    #####           USER INPUT          #####
    # Nazwa katalogu logów (np. "resnet50_standard_ce_multi_2025-06-26_11-16")
    log_dir_name = "resnet50_standard_ce_multi_2025-06-26_11-16"

    # Wyznacz katalog główny
    project_root = Path(__file__).resolve().parent.parent.parent

    # Ustal ścieżkę do katalogu logów na podstawie nazwy
    # Zakładamy, że katalog logów jest w project_root / "results" / "logs" / log_dir_name
    # To upraszcza potrzebę ładowania globalnego configu tylko po to, by uzyskać ścieżkę do logów.
    log_dir = project_root / "results" / "logs" / log_dir_name
    if not log_dir.is_dir():
        # Alternatywnie, można spróbować użyć PathManagera z globalnym configiem, jeśli powyższe zawiedzie
        # Ale to komplikuje. Na razie załóżmy, że struktura jest znana lub log_dir_name jest pełniejszą ścieżką.
        print(f"OSTRZEŻENIE: Katalog logów {log_dir} nie znaleziony przy założeniu struktury results/logs/. Spróbuję z globalnym configiem.")
        # Fallback do starej metody, jeśli potrzebne, ale lepiej tego unikać.
        # Na razie rzućmy błąd, jeśli nie ma katalogu.
        raise FileNotFoundError(f"Katalog logów {log_dir} nie został znaleziony.")

    print(f"Używam log_dir: {log_dir}")

    # Spróbuj załadować params.yaml z log_dir
    params_yaml_path = log_dir / "params.yaml"
    config_to_use_for_report_path = None # To będzie ścieżka do pliku config (params.yaml lub globalny)
    cfg_for_report_obj = None # To będzie załadowany obiekt OmegaConf

    if params_yaml_path.exists():
        print(f"Znaleziono params.yaml w katalogu logów: {params_yaml_path}")
        cfg_for_report_obj = OmegaConf.load(params_yaml_path)
        config_to_use_for_report_path = params_yaml_path
        print("Używam konfiguracji z params.yaml (zawartej w raporcie).")
    else:
        print(f"Nie znaleziono params.yaml w {log_dir}. Używam globalnego config.yaml.")
        global_config_path = project_root / "src" / "config" / "config.yaml"
        if not global_config_path.exists():
            raise FileNotFoundError(f"Globalny plik konfiguracyjny {global_config_path} nie znaleziony.")
        cfg_for_report_obj = OmegaConf.load(global_config_path)
        config_to_use_for_report_path = global_config_path

    if cfg_for_report_obj is None: # Powinno być już obsłużone przez FileNotFoundError powyżej
        raise RuntimeError("Nie udało się załadować żadnej konfiguracji.")

    # Utwórz PathManager z właściwą konfiguracją (tą, która będzie użyta w raporcie)
    path_manager = PathManager(project_root, cfg_for_report_obj)
    metadata_path = path_manager.metadata_file()

    print(f"Plik metadata użyty w raporcie (zgodnie z wybraną konfiguracją): {metadata_path}")
    print(f"Plik konfiguracyjny użyty w raporcie: {config_to_use_for_report_path}")

    # Wyszukaj wymagane pliki w log_dir
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
        config_path=config_to_use_for_report_path, # Użyj ścieżki do params.yaml lub globalnego configu
        predictions_path=predictions_path,
        metadata_path=metadata_path, # Ta ścieżka jest już zgodna z cfg_for_report_obj dzięki nowemu PathManagerowi
        metrics_path=metrics_path,
        augmentation_path=augmentation_path
    )
    report.run()

if __name__ == "__main__":
    main()
