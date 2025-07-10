import pytest
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import csv
import os
import sys # Potrzebne do sys.path, chociaz conftest powinien to załatwić

# Importy z projektu - dzięki add_src_to_sys_path w conftest, powinny działać
from engine.trainer import Trainer
from utils.path_manager import PathManager


# Fixture unique_test_run_artifacts_dir jest automatycznie wstrzykiwana przez pytest
# tak samo test_config_file_path i project_root_path

def modify_config_for_test_run(base_cfg: DictConfig, output_dir: Path, checkpoint_subdir_name: str) -> DictConfig:
    """Helper function to modify config paths to point to the unique test output directory."""
    cfg = base_cfg.copy() # Pracuj na kopii
    OmegaConf.set_struct(cfg, True) # Pozwól na zmiany struktury

    # PathManager będzie używał project_root do danych wejściowych.
    # Chcemy, aby logi i checkpointy trafiały do output_dir.
    # Kluczowe jest to, jak Trainer i PathManager interpretują te ścieżki.
    # Najprościej jest, jeśli PathManager może mieć osobny "output_root".
    # Bez tego, musimy użyć monkeypatching (jak zrobiono).

    # W config_test.yaml, checkpoint_dir jest np. "test_artifacts/checkpoints"
    # Chcemy, aby finalnie było to output_dir / "nazwa_z_config" -> output_dir / checkpoint_subdir_name
    cfg.training.checkpoint_dir = checkpoint_subdir_name

    # Inne ścieżki wyjściowe, jeśli są w cfg, też można by tu nadpisać,
    # np. cfg.prediction.results_dir, cfg.visualization.output_dir itp.
    # Na razie skupiamy się na logach i checkpointach, które są obsługiwane przez monkeypatching.
    OmegaConf.set_struct(cfg, False) # Zablokuj strukturę po modyfikacjach
    return cfg

def run_trainer_and_get_log_dir(project_root: Path, config_to_use: DictConfig, temp_cfg_file_path: Path) -> Path:
    """Runs the trainer and returns the specific log directory for the run."""
    OmegaConf.save(config_to_use, temp_cfg_file_path) # Zapisz config, który Trainer załaduje

    trainer = Trainer(
        project_root=project_root,
        config_path_override=str(temp_cfg_file_path),
        debug_mode=True # Wymuszone dla testów, nawet jeśli config mówi inaczej
    )
    trainer.train()

    # Znajdź katalog logów (powinien być jeden)
    # Zakładamy, że patched_logs_dir w conftest kieruje do unique_test_run_artifacts_dir/results/logs
    # a wewnątrz niego jest katalog przebiegu
    # Ta funkcja jest wywoływana PO monkeypatchingu PathManagera

    # Musimy odtworzyć ścieżkę, którą zwróciłby załatany logs_dir()
    # logs_dir() z PathManagera (załatanego) będzie wskazywać na coś w stylu:
    # unique_test_run_artifacts_dir / "results" / "logs"
    # A potem trainer_setup tworzy podkatalog z modelem, loss, timestamp.

    # Zamiast zgadywać, polegajmy na tym, że trainer.log_dir jest poprawnie ustawiony
    # przez trainer_setup, który używa załatanego PathManagera.
    assert hasattr(trainer, 'log_dir') and trainer.log_dir is not None, "trainer.log_dir nie został ustawiony"
    specific_log_dir = Path(trainer.log_dir)
    assert specific_log_dir.is_dir(), f"Katalog logów {specific_log_dir} nie istnieje lub nie jest katalogiem."
    return specific_log_dir


def verify_params_yaml(params_yaml_file: Path, expected_model_name: str, expected_model_mode: str, expected_loss: str, expect_weights: bool):
    assert params_yaml_file.exists(), f"Plik params.yaml nie znaleziony w {params_yaml_file.parent}"
    params_content = OmegaConf.load(params_yaml_file)
    assert params_content.model_name_used == expected_model_name, f"Oczekiwano modelu {expected_model_name}, jest {params_content.model_name_used}"
    assert params_content.model_mode == expected_model_mode, f"Oczekiwano trybu {expected_model_mode}, jest {params_content.model_mode}"
    assert params_content.loss_function_used == expected_loss, f"Oczekiwano funkcji straty {expected_loss}, jest {params_content.loss_function_used}"
    assert hasattr(params_content, 'run_timestamp')
    if expect_weights:
        assert hasattr(params_content, 'composite_score_weights'), "Brak wag composite_score w params.yaml dla trybu multitask"
        assert hasattr(params_content.composite_score_weights, 'alpha')
    else:
        assert not hasattr(params_content, 'composite_score_weights'), "Nie oczekiwano wag composite_score w params.yaml dla tego trybu"

def verify_metrics_csv(metrics_csv_file: Path, is_multitask: bool):
    assert metrics_csv_file.exists(), f"Plik metryk CSV nie znaleziony w {metrics_csv_file.parent}"
    with open(metrics_csv_file, 'r') as f:
        reader = csv.DictReader(f) # Użyj DictReader dla łatwiejszego dostępu po nazwie kolumny
        header = reader.fieldnames
        expected_new_metrics_cols = ['Val MAE Age', 'Val F1 Pop2 Age3-6', 'Val Composite Score']
        for col in expected_new_metrics_cols:
            assert col in header, f"Brak kolumny '{col}' w pliku metryk CSV."

        data_rows = list(reader)
        assert len(data_rows) >= 1, "Brak danych (epok) w pliku metryk CSV."

        first_epoch_data = data_rows[0]
        # Sprawdź, czy kluczowe metryki są obecne i mają wartości (nawet jeśli 'nan')
        for col in expected_new_metrics_cols:
            assert col in first_epoch_data, f"Brak kolumny '{col}' w danych pierwszej epoki."
            assert first_epoch_data[col] is not None, f"Wartość dla '{col}' to None w pierwszej epoce."

        if is_multitask:
            # Dla multitask, oczekujemy, że Composite Score może być liczbą (jeśli F1 global nie jest NaN)
            # MAE i F1 Subgroup mogą być liczbami lub NaN, ale nie powinny być puste jeśli kolumna istnieje
            assert first_epoch_data['Val MAE Age'] != ''
            assert first_epoch_data['Val F1 Pop2 Age3-6'] != ''
            assert first_epoch_data['Val Composite Score'] != '' # Może być 'nan' lub liczba
        else: # Base model
            assert first_epoch_data['Val MAE Age'].lower() == 'nan' or first_epoch_data['Val MAE Age'] == '', "MAE powinno być NaN dla base model"
            assert first_epoch_data['Val F1 Pop2 Age3-6'].lower() == 'nan' or first_epoch_data['Val F1 Pop2 Age3-6'] == '', "F1 Subgroup powinno być NaN dla base model"
            assert first_epoch_data['Val Composite Score'].lower() == 'nan' or first_epoch_data['Val Composite Score'] == '', "Composite Score powinno być NaN dla base model"


def verify_checkpoint(checkpoint_run_dir: Path, cfg_training: DictConfig, metrics_first_epoch_data: dict, is_multitask: bool):
    assert checkpoint_run_dir.exists(), f"Katalog checkpointów dla przebiegu {checkpoint_run_dir} nie istnieje."
    checkpoint_files = list(checkpoint_run_dir.glob("*.pth"))

    composite_score_val_str = metrics_first_epoch_data.get('Val Composite Score', 'nan')
    is_composite_score_nan = composite_score_val_str.lower() == 'nan' or composite_score_val_str == ''

    if is_multitask and not is_composite_score_nan:
        assert len(checkpoint_files) >= 1, "Nie zapisano pliku checkpointu dla multitask z poprawnym composite score."
        assert any("SCORE" in f.name for f in checkpoint_files), "Nazwa checkpointu dla multitask nie zawiera 'SCORE'."
    elif not is_multitask and is_composite_score_nan: # Base model, composite score jest NaN
        assert len(checkpoint_files) == 0, \
            f"Nie oczekiwano zapisu checkpointu dla base_model (Composite Score NaN), ale znaleziono: {checkpoint_files}"
    elif is_multitask and is_composite_score_nan: # Multitask, ale composite score wyszedł NaN
         assert len(checkpoint_files) == 0, \
            f"Nie oczekiwano zapisu checkpointu dla multitask (Composite Score NaN), ale znaleziono: {checkpoint_files}"
    else:
        # Inne przypadki, np. base model z nie-NaN composite score (niemożliwe przy obecnej logice)
        # lub jeśli logika zapisu się zmieni. Na razie ten warunek nie powinien być trafiony.
        pass


def test_training_run_multitask_debug_mode(
    project_root_path: Path,
    test_config_file_path: Path,
    unique_test_run_artifacts_dir: Path,
    monkeypatch
):
    base_cfg = OmegaConf.load(test_config_file_path)

    # Skonfiguruj dla multitask
    cfg_multitask = base_cfg.copy()
    OmegaConf.set_struct(cfg_multitask, True)
    cfg_multitask.multitask_model.use = True
    # Upewnij się, że base_model nie jest używany, jeśli multitask jest true
    # (chociaż logika Trainera powinna to obsłużyć)
    # cfg_multitask.base_model.use = False # Zakładając, że taki klucz by istniał

    # Ustaw nazwę podkatalogu dla checkpointów tego testu
    checkpoint_subdir_name = "cps_multitask"
    cfg_multitask.training.checkpoint_dir = checkpoint_subdir_name
    OmegaConf.set_struct(cfg_multitask, False)

    # Przygotuj ścieżkę do tymczasowego pliku config, który załaduje Trainer
    temp_cfg_file_for_run = unique_test_run_artifacts_dir / "config_multitask_run.yaml"

    # Monkeypatch PathManager
    # Chcemy, aby metody zwracały ścieżki WZGLĘDEM unique_test_run_artifacts_dir
    # dla logów i roota checkpointów.
    # `checkpoint_dir` w cfg to tylko nazwa podkatalogu wewnątrz załatanego roota checkpointów.

    # Patch logs_dir to return: unique_test_run_artifacts_dir / "results" / "logs"
    monkeypatch.setattr(PathManager, "logs_dir", lambda self: unique_test_run_artifacts_dir / "results" / "logs")
    # Patch checkpoint_dir to return: unique_test_run_artifacts_dir (root dla cfg.training.checkpoint_dir)
    monkeypatch.setattr(PathManager, "checkpoint_dir", lambda self: unique_test_run_artifacts_dir)

    # Uruchomienie treningu
    specific_log_dir = run_trainer_and_get_log_dir(project_root_path, cfg_multitask, temp_cfg_file_for_run)

    # Asercje
    verify_params_yaml(
        specific_log_dir / "params.yaml",
        expected_model_name=cfg_multitask.multitask_model.backbone_model.model_name,
        expected_model_mode="multitask",
        expected_loss=cfg_multitask.training.loss_type[0],
        expect_weights=True
    )

    metrics_csv_files = list(specific_log_dir.glob("*_training_metrics.csv"))
    assert len(metrics_csv_files) == 1, "Nie znaleziono pliku metryk CSV lub jest ich za dużo."
    metrics_csv_file = metrics_csv_files[0]
    verify_metrics_csv(metrics_csv_file, is_multitask=True)

    # Odczytaj dane z pierwszej epoki do sprawdzenia checkpointu
    with open(metrics_csv_file, 'r') as f:
        first_epoch_data = list(csv.DictReader(f))[0]

    # Checkpoint dir: unique_test_run_artifacts_dir / <cfg.training.checkpoint_dir> / <nazwa_katalogu_logu>
    # cfg_multitask.training.checkpoint_dir to "cps_multitask"
    # specific_log_dir.name to dynamiczna nazwa katalogu logu
    checkpoint_run_dir = unique_test_run_artifacts_dir / cfg_multitask.training.checkpoint_dir / specific_log_dir.name
    verify_checkpoint(checkpoint_run_dir, cfg_multitask.training, first_epoch_data, is_multitask=True)


def test_training_run_base_model_debug_mode(
    project_root_path: Path,
    test_config_file_path: Path,
    unique_test_run_artifacts_dir: Path,
    monkeypatch
):
    base_cfg = OmegaConf.load(test_config_file_path)

    cfg_base_model = base_cfg.copy()
    OmegaConf.set_struct(cfg_base_model, True)
    cfg_base_model.multitask_model.use = False # Kluczowa zmiana
    if not hasattr(cfg_base_model, 'base_model') or cfg_base_model.base_model is None: # Upewnij się, że sekcja istnieje
        cfg_base_model.base_model = OmegaConf.create({'model_name': 'resnet50', 'pretrained': False, 'image_size': 224}) # Domyślne wartości
    else: # Jeśli istnieje, upewnij się o kluczowych polach
        if not hasattr(cfg_base_model.base_model, 'model_name'): cfg_base_model.base_model.model_name = "resnet50"
        cfg_base_model.base_model.pretrained = False # Dla testów
        if not hasattr(cfg_base_model.base_model, 'image_size'): cfg_base_model.base_model.image_size = 224


    checkpoint_subdir_name = "cps_base"
    cfg_base_model.training.checkpoint_dir = checkpoint_subdir_name
    OmegaConf.set_struct(cfg_base_model, False)

    temp_cfg_file_for_run = unique_test_run_artifacts_dir / "config_base_run.yaml"

    monkeypatch.setattr(PathManager, "logs_dir", lambda self: unique_test_run_artifacts_dir / "results" / "logs")
    monkeypatch.setattr(PathManager, "checkpoint_dir", lambda self: unique_test_run_artifacts_dir)

    specific_log_dir = run_trainer_and_get_log_dir(project_root_path, cfg_base_model, temp_cfg_file_for_run)

    verify_params_yaml(
        specific_log_dir / "params.yaml",
        expected_model_name=cfg_base_model.base_model.model_name,
        expected_model_mode="base",
        expected_loss=cfg_base_model.training.loss_type[0],
        expect_weights=False
    )

    metrics_csv_files = list(specific_log_dir.glob("*_training_metrics.csv"))
    assert len(metrics_csv_files) == 1
    metrics_csv_file = metrics_csv_files[0]
    verify_metrics_csv(metrics_csv_file, is_multitask=False)

    with open(metrics_csv_file, 'r') as f:
        first_epoch_data = list(csv.DictReader(f))[0]

    checkpoint_run_dir = unique_test_run_artifacts_dir / cfg_base_model.training.checkpoint_dir / specific_log_dir.name
    verify_checkpoint(checkpoint_run_dir, cfg_base_model.training, first_epoch_data, is_multitask=False)

# TODO: Dodać testy dla metrics_calculation.py
# TODO: Dodać testy dla reporting.py
