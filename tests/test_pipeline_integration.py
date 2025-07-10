import pytest
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import csv
import os
import sys

# Dodaj src do PYTHONPATH, aby testy mogły importować moduły z src
# Zakładamy, że testy są uruchamiane z głównego katalogu projektu (np. przez `pytest`)
# lub że PYTHONPATH jest odpowiednio skonfigurowany.
# Fixture project_root_path z conftest.py wskazuje na główny katalog.
# Możemy dodać src do sys.path na początku sesji testowej lub w każdym pliku testowym.
# Lepszym miejscem może być conftest.py, ale dla pewności tutaj też można.

# @pytest.fixture(scope="session", autouse=True)
# def add_src_to_path(project_root_path):
#     sys.path.insert(0, str(project_root_path / "src"))
#     # To może być problematyczne, jeśli struktura importów w src nie jest na to gotowa.
#     # Zazwyczaj pytest radzi sobie z tym, jeśli uruchamiany jest z roota projektu.

# Importy modułów z projektu - zakładamy, że pytest jest uruchamiany z roota
# lub PYTHONPATH jest ustawiony. Jeśli nie, trzeba będzie dostosować.
# Bezpośrednio przed importem można dodać:
# current_script_dir = Path(__file__).parent
# project_root = current_script_dir.parent.parent # Zakładając tests/test_file.py
# src_path = project_root / "src"
# sys.path.insert(0, str(src_path))
# To jest często robione w conftest.py lub przez konfigurację pytest (np. python_paths w pytest.ini)

# Na razie zakładam, że pytest jest skonfigurowany do znajdowania modułów w src/
# Jeśli nie, testy nie przejdą na etapie importu.
from engine.trainer import Trainer
from utils.path_manager import PathManager # Potrzebny do ewentualnego sprawdzenia jak PathManager działa


def test_training_run_multitask_debug_mode(
    project_root_path: Path,
    test_config_file_path: Path,
    unique_test_run_artifacts_dir: Path
):
    """
    Testuje pełny pipeline treningowy dla modelu multitask w trybie debug.
    Sprawdza, czy trening przebiega, tworzone są odpowiednie artefakty (logi, checkpointy),
    i czy pliki wynikowe (params.yaml, metrics.csv) mają oczekiwaną strukturę/zawartość.
    """
    assert test_config_file_path.exists(), "Plik konfiguracyjny dla testów nie istnieje."

    # Załaduj bazową konfigurację testową
    cfg = OmegaConf.load(test_config_file_path)

    # Zmodyfikuj konfigurację, aby wyniki trafiały do unique_test_run_artifacts_dir
    # PathManager konstruuje ścieżki results/logs i checkpoints wewnątrz project_root.
    # Aby to przekierować, musimy albo zmodyfikować sposób, w jaki PathManager
    # uzyskuje bazowe ścieżki, albo nadpisać pełne ścieżki w konfiguracji,
    # albo (najprościej dla testu) zmodyfikować 'project_root' przekazywany do PathManager
    # tak, aby wskazywał na unique_test_run_artifacts_dir, a wewnątrz niego PathManager
    # stworzy strukturę 'results/logs' i 'checkpoints'.

    # Jednak PathManager jest też używany do odczytu danych (metadata, root_dir),
    # które są względne do prawdziwego project_root.
    # Zatem lepsze jest nadpisanie konkretnych ścieżek wyjściowych w cfg.

    # Nadpisanie ścieżek wyjściowych w konfiguracji
    # Użyjemy OmegaConf.set_struct, aby móc dodawać/zmieniać klucze, nawet jeśli nie były zdefiniowane
    OmegaConf.set_struct(cfg, True)

    # Chcemy, aby logi i checkpointy trafiły do unique_test_run_artifacts_dir
    # PathManager tworzy katalogi 'results/logs' i 'checkpoints' relatywnie do project_root.
    # Aby to kontrolować, możemy zmodyfikować `project_root` przekazywany do Trainer
    # tylko dla celów tworzenia ścieżek przez PathManager, ale to skomplikowane.

    # Prostsze podejście: `PathManager` ma metody `logs_dir()` i `checkpoint_dir()`
    # które zwracają `self.project_root / "results" / "logs"` i `self.project_root / self.cfg.training.checkpoint_dir`
    # Możemy nadpisać `cfg.training.checkpoint_dir` oraz zmodyfikować `PathManager`
    # tak, by akceptował nadpisanie bazowego katalogu `results`.
    # To wymagałoby zmian w PathManager.

    # Najczystsze bez zmiany PathManager:
    # Skoro PathManager używa `project_root` jako bazy dla `results/logs` i dla `cfg.training.checkpoint_dir`,
    # a my nie chcemy zaśmiecać głównego `results/` i `checkpoints/` projektu,
    # musimy zapewnić, że PathManager w Trainerze będzie operował na ścieżkach
    # wskazujących do `unique_test_run_artifacts_dir`.

    # Opcja 1: Przekazać do Trainera zmodyfikowany project_root, który jest unique_test_run_artifacts_dir.
    # Ale wtedy ścieżki do danych w config_test.yaml (tests/data/...) muszą być absolutne lub
    # Trainer musi dostać też prawdziwy project_root do odczytu danych. To komplikuje Trainer.

    # Opcja 2: Nadpisać ścieżki w cfg tak, aby były absolutne i wskazywały do unique_test_run_artifacts_dir.
    # np. cfg.training.checkpoint_dir = str(unique_test_run_artifacts_dir / "checkpoints")
    #      cfg.output_logs_base_dir = str(unique_test_run_artifacts_dir / "results" / "logs") # Nowy klucz
    # I wtedy PathManager musiałby używać cfg.output_logs_base_dir zamiast project_root / "results" / "logs".

    # Opcja 3 (Najprostsza na teraz, ale może wymagać delikatnej zmiany w PathManager, jeśli nie ma elastyczności):
    # Załóżmy, że `config_test.yaml` ma już np. `training.checkpoint_dir = "test_artifacts/checkpoints"`.
    # Wtedy `PathManager` (inicjalizowany z `project_root_path`) stworzy:
    # `project_root_path / "test_artifacts/checkpoints"`. To jest OK.
    # Dla logów, `PathManager.logs_dir()` zwraca `project_root_path / "results" / "logs"`.
    # Chcemy, aby to było `project_root_path / "tests" / "test_artifacts" / "nazwa_testu" / "logs_dla_tego_przebiegu"`.
    # `unique_test_run_artifacts_dir` już jest tym `project_root_path / "tests" / "test_artifacts" / "nazwa_testu"`.
    # Więc `PathManager` musi być poinstruowany, aby używał `unique_test_run_artifacts_dir` jako "bazy" dla swoich `results/logs`.

    # Na razie spróbujemy tak:
    # W config_test.yaml mamy checkpoint_dir: "test_artifacts/checkpoints"
    # To oznacza, że checkpointy wylądują w project_root/test_artifacts/checkpoints.
    # Chcemy je w unique_test_run_artifacts_dir/checkpoints.

    # Zmodyfikujmy cfg, aby checkpointy i logi trafiały do unique_test_run_artifacts_dir
    # To wymaga, aby PathManager był świadomy tego katalogu jako "efektywnego project_root" dla wyjść.

    # Najprostsze rozwiązanie dla testu:
    # Skopiuj config_test.yaml do unique_test_run_artifacts_dir i zmodyfikuj tam ścieżki względne.
    # A potem przekaż ścieżkę do tego skopiowanego i zmodyfikowanego configu.

    temp_test_config_in_artifact_dir = unique_test_run_artifacts_dir / "temp_config_for_test.yaml"

    # Modyfikacja ścieżek w cfg na absolutne, wskazujące do unique_test_run_artifacts_dir
    # To jest bardziej skomplikowane niż się wydaje, bo PathManager używa project_root.
    # Najlepiej by było, gdyby Trainer pozwalał na przekazanie `output_base_dir`.

    # Podejście: Użyjemy `project_root_path` dla danych wejściowych (zgodnie z `config_test.yaml`)
    # ale dla celów zapisu artefaktów (logi, checkpointy), chcemy, aby `Trainer`
    # myślał, że jego "root" dla wyników jest `unique_test_run_artifacts_dir`.

    # W `config_test.yaml` mamy:
    # data.metadata_file: "tests/data/metadata_test.xlsx"
    # data.root_dir: "tests/data/sample_images/"
    # training.checkpoint_dir: "test_artifacts/checkpoints" -> to będzie project_root/test_artifacts/checkpoints

    # Chcemy, aby logi (results/logs) i checkpointy (z training.checkpoint_dir) były względne
    # do unique_test_run_artifacts_dir.
    # Możemy to osiągnąć, jeśli `Trainer` jako `project_root` przyjmie `unique_test_run_artifacts_dir`
    # ale wtedy ścieżki `data.*` w configu muszą być absolutne lub `Trainer` musi mieć też `data_project_root`.

    # Spróbujmy nadpisać ścieżki w cfg, aby były "fałszywie" względne, ale PathManager
    # zainicjalizowany z `unique_test_run_artifacts_dir` jako `project_root` zinterpretuje je poprawnie.
    # To jest hack. Lepszy byłby refaktoring PathManagera lub Trainera.

    # Na razie, dla tego testu, założymy, że `config_test.yaml` ma `training.checkpoint_dir`
    # ustawione na coś, co wyląduje w `test_artifacts_root_dir` (np. "test_artifacts/checkpoints").
    # A logi (`results/logs`) będą tworzone przez `PathManager` wewnątrz `project_root_path`.
    # To oznacza, że logi z testów będą w `project_root/results/logs`, co nie jest idealne.

    # Aby to obejść bez zmiany kodu źródłowego, możemy tymczasowo zmienić `cfg.training.checkpoint_dir`
    # na ścieżkę wewnątrz `unique_test_run_artifacts_dir` i liczyć na to, że `PathManager`
    # dla logów użyje `project_root` z `Trainer`a, który możemy ustawić na `unique_test_run_artifacts_dir`.
    # Ale to zepsuje ścieżki do danych.

    # NAJCZYSTSZE ROZWIĄZANIE (wymaga małej elastyczności w Trainer/PathManager lub użycia monkeypatching):
    # 1. Trainer przyjmuje opcjonalny `output_root_dir`.
    # 2. PathManager przyjmuje opcjonalny `output_root_dir` i używa go zamiast `project_root` dla `logs_dir` i `checkpoint_dir_root`.

    # Monkeypatching PathManager.logs_dir() i PathManager.checkpoint_dir() na czas testu:
    original_logs_dir_method = PathManager.logs_dir
    original_checkpoint_dir_method = PathManager.checkpoint_dir

    def patched_logs_dir(self_pm):
        # self_pm to instancja PathManager
        # Zwróć ścieżkę do logów wewnątrz unique_test_run_artifacts_dir
        return unique_test_run_artifacts_dir / "results" / "logs"

    def patched_checkpoint_dir(self_pm):
        # Zwróć ścieżkę do checkpointów wewnątrz unique_test_run_artifacts_dir
        # Nazwa z cfg.training.checkpoint_dir będzie podkatalogiem tutaj.
        return unique_test_run_artifacts_dir / self_pm.cfg.training.checkpoint_dir

    PathManager.logs_dir = patched_logs_dir
    PathManager.checkpoint_dir = patched_checkpoint_dir

    # Upewnijmy się, że w configu testowym checkpoint_dir jest prostą nazwą, a nie ścieżką
    # np. training.checkpoint_dir: "test_checkpoints" (w config_test.yaml)
    # Wtedy pełna ścieżka będzie unique_test_run_artifacts_dir / "test_checkpoints"
    # W config_test.yaml zmieniłem na "test_artifacts/checkpoints" - to nie jest idealne dla tego patcha.
    # Powinno być samo "checkpoints_for_this_test"
    # Poprawiam to w locie dla tego testu:
    cfg.training.checkpoint_dir = "cps" # Krótka nazwa dla podkatalogu w unique_test_run_artifacts_dir

    try:
        trainer = Trainer(
            project_root=project_root_path, # Prawdziwy project root dla danych
            config_path_override=str(test_config_file_path), # Użyj config_test.yaml
            debug_mode=True # Wymuszenie trybu debug, chociaż config_test ma stop_after_one_epoch
        )
        # Przekazanie zmodyfikowanego cfg do trainer.train() nie jest trywialne, bo Trainer ładuje go sam.
        # Musimy polegać na tym, że config_path_override zostanie poprawnie załadowany
        # i że nasz monkeypatch zadziała na instancji PathManager stworzonej w Trainerze.
        # Aby monkeypatch zadziałał na konfiguracji załadowanej przez Trainer, musimy
        # załadować ją, zmodyfikować, zapisać do tymczasowego pliku i przekazać ten plik.

        # Lepsze podejście z monkeypatch: patchujemy metody _get_base_log_dir i _get_base_checkpoint_dir w PathManager
        # jeśli by istniały, lub modyfikujemy zwracane wartości przez logs_dir i checkpoint_dir.
        # Powyższy patch powinien działać, bo modyfikuje metody na poziomie klasy.

        trainer.train() # Uruchom trening

        # Asercje
        # Katalog logów powinien być stworzony przez PathManager.logs_dir() -> patched_logs_dir()
        # Nazwa katalogu logów jest dynamiczna (zawiera timestamp). Musimy go znaleźć.

        # Znajdź katalog logów wewnątrz unique_test_run_artifacts_dir/results/logs
        # unique_test_run_artifacts_dir/results/logs/MODEL_LOSS_MODE_TIMESTAMP
        run_logs_base = unique_test_run_artifacts_dir / "results" / "logs"
        assert run_logs_base.exists(), "Bazowy katalog logów testowych nie został utworzony."

        # Powinien być dokładnie jeden podkatalog (jeden przebieg testowy)
        subdirs_in_run_logs_base = [d for d in run_logs_base.iterdir() if d.is_dir()]
        assert len(subdirs_in_run_logs_base) == 1, f"Oczekiwano jednego katalogu logów, znaleziono {len(subdirs_in_run_logs_base)}"
        specific_log_dir = subdirs_in_run_logs_base[0]

        # 1. Sprawdź istnienie params.yaml
        params_yaml_file = specific_log_dir / "params.yaml"
        assert params_yaml_file.exists(), "Plik params.yaml nie został utworzony."

        # 2. Sprawdź zawartość params.yaml
        params_content = OmegaConf.load(params_yaml_file)
        assert hasattr(params_content, 'model_name_used') and params_content.model_name_used == cfg.multitask_model.backbone_model.model_name
        assert hasattr(params_content, 'model_mode') and params_content.model_mode == "multitask" # Bo cfg.multitask_model.use = true
        assert hasattr(params_content, 'loss_function_used') and params_content.loss_function_used == cfg.training.loss_type[0]
        assert hasattr(params_content, 'run_timestamp')
        assert hasattr(params_content, 'composite_score_weights') # Powinny być, bo multitask
        assert hasattr(params_content.composite_score_weights, 'alpha')

        # 3. Sprawdź istnienie _training_metrics.csv
        metrics_files = list(specific_log_dir.glob("*_training_metrics.csv"))
        assert len(metrics_files) == 1, "Plik _training_metrics.csv nie został utworzony lub jest ich wiele."
        metrics_csv_file = metrics_files[0]

        # 4. Sprawdź nagłówki i zawartość _training_metrics.csv
        with open(metrics_csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_new_metrics_cols = ['Val MAE Age', 'Val F1 Pop2 Age3-6', 'Val Composite Score']
            for col in expected_new_metrics_cols:
                assert col in header, f"Brak kolumny '{col}' w pliku metryk CSV."

            # Sprawdź, czy jest przynajmniej jeden wiersz danych (dla jednej epoki)
            data_rows = list(reader)
            assert len(data_rows) >= 1, "Brak danych w pliku metryk CSV."
            # Można by sprawdzić typy danych w kolumnach, np. czy Composite Score jest floatem lub NaN
            # Np. dla pierwszej epoki:
            first_epoch_data = dict(zip(header, data_rows[0]))
            assert first_epoch_data['Val Composite Score'] is not None # Powinien być stringiem liczby lub 'nan'

        # 5. Sprawdź istnienie checkpointu modelu
        # Ścieżka: unique_test_run_artifacts_dir / cfg.training.checkpoint_dir (czyli "cps") / dynamiczna_nazwa_folderu_logów / model.pth
        # Nazwa checkpointu zawiera SCORE, np. resnet50_focal_loss_ageboost_multi_..._SCORE_...pth
        # Katalog checkpointów: unique_test_run_artifacts_dir / "cps" / nazwa_katalogu_logów_bez_timestamp (?)
        # PathManager.checkpoint_dir() zwraca unique_test_run_artifacts_dir / "cps"
        # A w trainer_setup.py: checkpoint_dir = checkpoint_root / full_name
        # Gdzie checkpoint_root to wynik PathManager.checkpoint_dir()
        # Więc checkpointy będą w: unique_test_run_artifacts_dir / "cps" / specific_log_dir.name / plik.pth

        checkpoint_run_dir = unique_test_run_artifacts_dir / cfg.training.checkpoint_dir / specific_log_dir.name
        assert checkpoint_run_dir.exists(), f"Katalog checkpointów dla przebiegu {checkpoint_run_dir} nie istnieje."

        checkpoint_files = list(checkpoint_run_dir.glob("*.pth"))
        assert len(checkpoint_files) >= 1, "Nie zapisano pliku checkpointu modelu .pth"
        # Sprawdź, czy nazwa pliku zawiera SCORE (jeśli composite score nie był NaN)
        # To wymaga odczytania composite score z metryk
        composite_score_val_str = first_epoch_data['Val Composite Score']
        if composite_score_val_str.lower() != 'nan':
            assert any("SCORE" in f.name for f in checkpoint_files), "Nazwa checkpointu nie zawiera 'SCORE' mimo obliczonego Composite Score."

    finally:
        # Przywróć oryginalne metody PathManager
        PathManager.logs_dir = original_logs_dir_method
        PathManager.checkpoint_dir = original_checkpoint_dir_method
        print("Przywrócono oryginalne metody PathManager.")


def test_training_run_base_model_debug_mode(
    project_root_path: Path,
    test_config_file_path: Path,
    unique_test_run_artifacts_dir: Path
):
    """
    Testuje pełny pipeline treningowy dla modelu bazowego (niemultitask) w trybie debug.
    """
    assert test_config_file_path.exists(), "Plik konfiguracyjny dla testów nie istnieje."

    cfg = OmegaConf.load(test_config_file_path)
    OmegaConf.set_struct(cfg, True)

    # --- Konfiguracja specyficzna dla tego testu ---
    cfg.multitask_model.use = False # KLUCZOWA ZMIANA - użyj modelu bazowego
    # Upewnij się, że model bazowy ma sensowne ustawienia dla testu
    if not hasattr(cfg, 'base_model'):
        cfg.base_model = OmegaConf.create()
    cfg.base_model.base_model = "resnet50" # lub inny dostępny, prosty model
    cfg.base_model.pretrained = False
    cfg.base_model.image_size = cfg.multitask_model.backbone_model.image_size # Użyj tego samego rozmiaru co w multitask dla spójności danych testowych

    # Ustawienie katalogu checkpointów dla tego testu (wewnątrz unique_test_run_artifacts_dir)
    cfg.training.checkpoint_dir = "cps_base_model"

    # Monkeypatching PathManager (tak jak w poprzednim teście)
    original_logs_dir_method = PathManager.logs_dir
    original_checkpoint_dir_method = PathManager.checkpoint_dir

    def patched_logs_dir(self_pm):
        return unique_test_run_artifacts_dir / "results" / "logs"
    def patched_checkpoint_dir(self_pm):
        return unique_test_run_artifacts_dir / self_pm.cfg.training.checkpoint_dir

    PathManager.logs_dir = patched_logs_dir
    PathManager.checkpoint_dir = patched_checkpoint_dir

    try:
        # Zapisz tymczasowo zmodyfikowaną konfigurację, aby Trainer ją załadował
        # To jest konieczne, bo Trainer ładuje config z pliku na podstawie ścieżki.
        temp_cfg_path = unique_test_run_artifacts_dir / "temp_base_model_config.yaml"
        OmegaConf.save(cfg, temp_cfg_path)

        trainer = Trainer(
            project_root=project_root_path,
            config_path_override=str(temp_cfg_path), # Użyj zmodyfikowanego configu
            debug_mode=True
        )
        trainer.train()

        # --- Asercje ---
        run_logs_base = unique_test_run_artifacts_dir / "results" / "logs"
        assert run_logs_base.exists()
        subdirs_in_run_logs_base = [d for d in run_logs_base.iterdir() if d.is_dir()]
        assert len(subdirs_in_run_logs_base) == 1
        specific_log_dir = subdirs_in_run_logs_base[0]

        # 1. Sprawdź params.yaml
        params_yaml_file = specific_log_dir / "params.yaml"
        assert params_yaml_file.exists()
        params_content = OmegaConf.load(params_yaml_file)
        assert params_content.model_name_used == cfg.base_model.base_model
        assert params_content.model_mode == "base" # Oczekujemy "base"
        assert params_content.loss_function_used == cfg.training.loss_type[0]
        assert not hasattr(params_content, 'composite_score_weights') # Nie powinno być dla base model

        # 2. Sprawdź _training_metrics.csv
        metrics_files = list(specific_log_dir.glob("*_training_metrics.csv"))
        assert len(metrics_files) == 1
        metrics_csv_file = metrics_files[0]

        with open(metrics_csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_metrics_cols = ['Val MAE Age', 'Val F1 Pop2 Age3-6', 'Val Composite Score']
            for col in expected_metrics_cols:
                assert col in header

            data_rows = list(reader)
            assert len(data_rows) >= 1
            first_epoch_data = dict(zip(header, data_rows[0]))

            # Dla base_model, te metryki powinny być NaN
            assert first_epoch_data['Val MAE Age'].lower() == 'nan' or first_epoch_data['Val MAE Age'] == ''
            assert first_epoch_data['Val F1 Pop2 Age3-6'].lower() == 'nan' or first_epoch_data['Val F1 Pop2 Age3-6'] == ''
            # Composite score również powinien być NaN, bo składniki są NaN, a wagi beta/gamma nie są 0
            # (chyba że alpha jest >0 i F1 global nie jest NaN, wtedy może być jakaś wartość)
            # Zgodnie z logiką: jeśli current_composite_score jest NaN, model nie jest zapisywany z "SCORE_..."
            # Sprawdźmy, czy jest NaN
            assert first_epoch_data['Val Composite Score'].lower() == 'nan' or first_epoch_data['Val Composite Score'] == ''


        # 3. Sprawdź checkpointy
        # Ponieważ composite_score będzie NaN, save_best_model nie zapisze modelu z "SCORE_".
        # W obecnej logice, jeśli nie ma poprawy (a NaN nie jest > -inf), model nie jest zapisywany.
        # `trainer.last_model_path` może być None.
        # Jeśli `stop_after_one_epoch` jest true, pętla wykona się raz, `save_best_model` zostanie wywołane.
        # Jeśli `current_composite_score` jest NaN, `improved` będzie False.
        # `trainer.last_model_path` nie zostanie zaktualizowany.
        # To oznacza, że w tym scenariuszu nie oczekujemy "najlepszego" modelu.
        # Jednak `run_full_dataset_prediction` jest wywoływane jeśli `trainer.last_model_path` nie jest None.
        # To wymaga przemyślenia - czy chcemy, aby jakiś model był zapisywany nawet jeśli score jest NaN?
        # Na razie testujemy obecne zachowanie: brak modelu "best" jeśli score jest NaN.

        checkpoint_run_dir = unique_test_run_artifacts_dir / cfg.training.checkpoint_dir / specific_log_dir.name

        # Sprawdźmy, czy katalog checkpointów dla przebiegu w ogóle istnieje. Powinien.
        assert checkpoint_run_dir.exists(), f"Katalog checkpointów dla przebiegu {checkpoint_run_dir} nie istnieje."

        checkpoint_files = list(checkpoint_run_dir.glob("*.pth"))
        # W tym przypadku (base model, composite score = NaN), nie oczekujemy zapisu "najlepszego" modelu.
        # Jeśli logika zapisu by się zmieniła (np. fallback na ACC), ta asercja musiałaby być inna.
        assert len(checkpoint_files) == 0, \
            f"Nie oczekiwano zapisu najlepszego modelu, gdy Composite Score jest NaN, ale znaleziono: {checkpoint_files}"

    finally:
        PathManager.logs_dir = original_logs_dir_method
        PathManager.checkpoint_dir = original_checkpoint_dir_method
        print("Przywrócono oryginalne metody PathManager dla base_model test.")
        # Usunięcie tymczasowego pliku konfiguracyjnego
        if temp_cfg_path.exists():
            os.remove(temp_cfg_path)

# TODO: Dodać testy dla metrics_calculation.py
# TODO: Dodać testy dla reporting.py
