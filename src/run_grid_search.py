import sys
import os
from pathlib import Path
import itertools
import csv
from omegaconf import OmegaConf, DictConfig

# Dodanie src do PYTHONPATH, aby umożliwić importy z src/
# Zakładamy, że run_grid_search.py jest w src/
# Jeśli jest gdzie indziej, trzeba dostosować ścieżkę
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.engine.trainer import Trainer # Zakładając, że Trainer jest w src.engine.trainer
# Jeśli ścieżki są inne, dostosuj import:
# from engine.trainer import Trainer

def run_single_training(config_path: str, project_root: Path, current_weights: dict) -> dict:
    """
    Uruchamia pojedynczy cykl treningowy z danymi wagami i zwraca najlepsze metryki.
    """
    print(f"\n{'='*20} Rozpoczynanie treningu z wagami: {current_weights} {'='*20}")

    # Załaduj oryginalną konfigurację
    base_cfg = OmegaConf.load(project_root / config_path)

    # Stwórz kopię konfiguracji do modyfikacji lub modyfikuj bezpośrednio, jeśli Trainer tworzy głęboką kopię
    # Dla bezpieczeństwa, lepiej stworzyć nową konfigurację lub zmodyfikować kopię.
    # OmegaConf pozwala na scalanie konfiguracji.

    modified_cfg_dict = OmegaConf.to_container(base_cfg, resolve=True)

    # Nadpisz wagi w konfiguracji
    if 'multitask_model' not in modified_cfg_dict:
        modified_cfg_dict['multitask_model'] = {}
    if 'metrics_weights' not in modified_cfg_dict['multitask_model']:
        modified_cfg_dict['multitask_model']['metrics_weights'] = {}

    modified_cfg_dict['multitask_model']['metrics_weights']['alpha'] = current_weights['alpha']
    modified_cfg_dict['multitask_model']['metrics_weights']['beta'] = current_weights['beta']
    modified_cfg_dict['multitask_model']['metrics_weights']['gamma'] = current_weights['gamma']

    # Stwórz tymczasowy plik konfiguracyjny lub przekaż DictConfig do Trainer
    # Przekazanie DictConfig jest czystsze, jeśli Trainer to obsługuje.
    # Obecnie Trainer w __init__ ładuje config z config_path.
    # Można by zmodyfikować Trainer, aby opcjonalnie przyjmował DictConfig,
    # lub zapisać tymczasowy plik YAML.
    # Na razie zapiszemy tymczasowy plik YAML.

    temp_config_path = project_root / "src/config/temp_config_grid_search.yaml"
    temp_cfg_to_save = OmegaConf.create(modified_cfg_dict)
    OmegaConf.save(config=temp_cfg_to_save, f=str(temp_config_path))

    best_metrics_for_run = {
        "alpha": current_weights['alpha'],
        "beta": current_weights['beta'],
        "gamma": current_weights['gamma'],
        "best_composite_score": -float('inf'),
        "best_f1_global": -float('inf'),
        "best_mae_age": float('inf'), # Chcemy minimalizować MAE
        "best_f1_subgroup": -float('inf'),
        "log_dir": None
    }

    try:
        trainer = Trainer(config_path=str(temp_config_path), project_root=project_root)
        trainer.train() # Metoda train powinna wykonać cały trening

        # Po treningu, musimy uzyskać najlepszy composite_score
        # `trainer.best_score` przechowuje najlepszy composite_score z tego przebiegu
        if hasattr(trainer, 'best_score') and trainer.best_score > -float('inf'):
            best_metrics_for_run["best_composite_score"] = trainer.best_score

            # Aby uzyskać odpowiadające F1, MAE, F1_subgroup, musielibyśmy je zapisać
            # razem z best_score w obiekcie trainer, gdy model jest zapisywany.
            # Alternatywnie, możemy odczytać ostatnią linię z pliku CSV metryk dla tego przebiegu.
            # To drugie podejście jest bardziej niezawodne, jeśli `trainer` nie przechowuje tych szczegółów.
            # Log dir jest w trainer.log_dir
            if hasattr(trainer, 'log_dir') and trainer.log_dir:
                metrics_csv_path = None
                for f_name in os.listdir(trainer.log_dir):
                    if f_name.endswith("_training_metrics.csv"):
                        metrics_csv_path = trainer.log_dir / f_name
                        break

                if metrics_csv_path and metrics_csv_path.exists():
                    best_metrics_for_run["log_dir"] = str(trainer.log_dir)
                    with open(metrics_csv_path, 'r') as f:
                        reader = csv.DictReader(f)
                        last_best_epoch_metrics = None
                        current_best_run_score = -float('inf')
                        for row in reader:
                            # Szukamy epoki z najlepszym composite_score dla TEGO przebiegu
                            # (trainer.best_score to wartość z tej epoki)
                            # Kolumny: 'Val Composite Score', 'Val F1', 'Val MAE Age', 'Val F1 Pop2 Age3-6'
                            try:
                                score_val = float(row.get('Val Composite Score', np.nan))
                                # Użyj np.isclose do porównywania floatów
                                if not np.isnan(score_val) and hasattr(trainer, 'best_score') and np.isclose(score_val, trainer.best_score):
                                    last_best_epoch_metrics = row
                                    # Teoretycznie powinna być tylko jedna taka epoka, ale na wszelki wypadek
                                    # bierzemy ostatnią, jeśli byłoby ich więcej z tym samym score.
                                    # Lub można by brać pierwszą.
                                    break # Znaleziono epokę z najlepszym wynikiem
                                elif not np.isnan(score_val) and score_val > current_best_run_score:
                                    # Jeśli z jakiegoś powodu trainer.best_score nie jest idealnie zsynchronizowany,
                                    # to awaryjnie szukamy najlepszego w pliku.
                                    current_best_run_score = score_val
                                    last_best_epoch_metrics = row


                            except ValueError:
                                continue # Pomiń wiersze, gdzie konwersja na float się nie udaje

                        if last_best_epoch_metrics:
                            best_metrics_for_run["best_f1_global"] = float(last_best_epoch_metrics.get('Val F1', np.nan))
                            best_metrics_for_run["best_mae_age"] = float(last_best_epoch_metrics.get('Val MAE Age', np.nan))
                            best_metrics_for_run["best_f1_subgroup"] = float(last_best_epoch_metrics.get('Val F1 Pop2 Age3-6', np.nan))
                            # Upewnijmy się, że composite score jest ten sam
                            best_metrics_for_run["best_composite_score"] = float(last_best_epoch_metrics.get('Val Composite Score', np.nan))

        print(f"Wyniki dla wag {current_weights}: {best_metrics_for_run}")

    except Exception as e:
        print(f"Błąd podczas treningu z wagami {current_weights}: {e}")
        # Zapisz błąd lub wartości NaN, aby zaznaczyć problem
        best_metrics_for_run["best_composite_score"] = np.nan
        # Można dodać pole "error_message"
    finally:
        if temp_config_path.exists():
            os.remove(temp_config_path) # Usuń tymczasowy plik konfiguracyjny
            print(f"Usunięto tymczasowy plik konfiguracyjny: {temp_config_path}")

    return best_metrics_for_run


def main():
    project_root = Path(__file__).resolve().parent.parent
    default_config_path = "src/config/config.yaml" # Względna do project_root

    # Zdefiniuj zakresy wag do przeszukania
    # Np. alpha_values = [0.2, 0.4, 0.6, 0.8]
    # Aby suma wag była 1, można generować je inaczej, np.
    # (a,b,c) takie, że a+b+c=1
    # Dla uproszczenia, na razie zdefiniujmy kilka kombinacji ręcznie
    # lub użyjmy itertools.product dla siatki.

    # Przykład: siatka wartości
    alpha_values = [0.2, 0.5, 0.8]
    beta_values = [0.2, 0.5, 0.8]
    # gamma będzie dopełnieniem do 1, lub też z siatki, a potem normalizujemy

    # Generowanie kombinacji, gdzie suma wag = 1
    weight_combinations = []
    # Można użyć np. kroku 0.1 lub 0.2
    steps = np.arange(0.1, 1.0, 0.2) # np. [0.1, 0.3, 0.5, 0.7, 0.9]
    for alpha in steps:
        for beta in steps:
            if alpha + beta < 1.0: # Gamma musi być > 0
                gamma = 1.0 - alpha - beta
                if gamma > 0: # Upewnij się, że gamma jest dodatnia
                     # Zaokrąglenie, aby uniknąć problemów z precyzją float
                    weight_combinations.append({
                        "alpha": round(alpha, 2),
                        "beta": round(beta, 2),
                        "gamma": round(gamma, 2)
                    })
            # Można też dodać przypadki brzegowe, np. (1,0,0), (0,1,0), (0,0,1)

    # Dodaj przypadki brzegowe dla pewności
    weight_combinations.extend([
        {"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
        {"alpha": 0.0, "beta": 1.0, "gamma": 0.0},
        {"alpha": 0.0, "beta": 0.0, "gamma": 1.0},
        {"alpha": 0.33, "beta": 0.33, "gamma": 0.34}, # Domyślne
    ])
    # Usuń duplikaty, jeśli jakieś powstały
    unique_combinations = []
    seen_tuples = set()
    for wc in weight_combinations:
        # Tworzenie krotki do sprawdzania duplikatów, aby zachować kolejność kluczy
        # (choć dla dict nie ma to znaczenia, ale dla setu tak)
        # Lepiej sortować klucze, aby (a:1, b:2) było tym samym co (b:2, a:1)
        # Ale tu klucze są stałe ('alpha', 'beta', 'gamma')
        # Po prostu użyjemy wartości

        # Zaokrąglanie wag, aby uniknąć problemów z precyzją float przy porównywaniu
        # i sumowaniu do 1.0.
        wc_tuple = (round(wc['alpha'],3), round(wc['beta'],3), round(wc['gamma'],3))
        if wc_tuple not in seen_tuples and np.isclose(sum(wc_tuple), 1.0):
            unique_combinations.append(wc)
            seen_tuples.add(wc_tuple)

    weight_combinations = unique_combinations
    print(f"Przetestowane kombinacje wag ({len(weight_combinations)}): {weight_combinations}")

    results_file_path = project_root / "grid_search_results.csv"

    # Nagłówek pliku CSV
    csv_header = [
        "alpha", "beta", "gamma",
        "best_composite_score", "best_f1_global",
        "best_mae_age", "best_f1_subgroup", "log_dir"
    ]

    with open(results_file_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        writer.writeheader()

        all_run_results = []
        for weights in weight_combinations:
            run_metrics = run_single_training(default_config_path, project_root, weights)
            writer.writerow(run_metrics)
            all_run_results.append(run_metrics)
            f.flush() # Zapisz na bieżąco

    # Znajdź najlepszą kombinację
    if all_run_results:
        # Filtruj wyniki, gdzie composite_score nie jest NaN
        valid_results = [r for r in all_run_results if not np.isnan(r.get("best_composite_score", np.nan))]
        if valid_results:
            best_run = max(valid_results, key=lambda x: x["best_composite_score"])
            print("\n" + "="*30 + " Grid Search Zakończony " + "="*30)
            print(f"Najlepsza kombinacja wag: Alpha={best_run['alpha']}, Beta={best_run['beta']}, Gamma={best_run['gamma']}")
            print(f"Najlepszy Composite Score: {best_run['best_composite_score']:.4f}")
            print(f"  Odpowiadający F1 Global: {best_run['best_f1_global']:.4f}")
            print(f"  Odpowiadający MAE Age: {best_run['best_mae_age']:.4f}")
            print(f"  Odpowiadający F1 Subgroup: {best_run['best_f1_subgroup']:.4f}")
            print(f"  Logi dla najlepszego przebiegu: {best_run['log_dir']}")
            print(f"Pełne wyniki zapisano w: {results_file_path}")
        else:
            print("\n" + "="*30 + " Grid Search Zakończony " + "="*30)
            print("Nie znaleziono żadnych prawidłowych wyników (wszystkie composite_score były NaN).")
            print(f"Wyniki (prawdopodobnie z błędami) zapisano w: {results_file_path}")
    else:
        print("Grid Search nie wygenerował żadnych wyników.")

if __name__ == "__main__":
    # Upewnij się, że importy z src działają.
    # Jeśli uruchamiasz z katalogu głównego projektu: python src/run_grid_search.py
    # Jeśli uruchamiasz z src: python run_grid_search.py

    # Poprawka ścieżki, aby działało niezależnie od miejsca uruchomienia
    current_script_path = Path(__file__).resolve()
    project_root_for_imports = current_script_path.parent.parent # Zakładając, że skrypt jest w src/

    # Sprawdzenie, czy src jest już w PYTHONPATH, jeśli nie, dodaj katalog nadrzędny src
    # (czyli root projektu)
    if str(project_root_for_imports) not in sys.path:
         sys.path.insert(0, str(project_root_for_imports))

    # Ponowny import Trainer, na wypadek gdyby pierwszy sys.path.append nie zadziałał poprawnie
    # w niektórych scenariuszach uruchomienia.
    # Lepszym rozwiązaniem byłoby użycie `python -m src.run_grid_search` z katalogu projektu.
    # Ale dla prostoty skryptu, spróbujmy tak:
    try:
        from engine.trainer import Trainer # Bez src. na początku, bo src/parent jest w path
    except ImportError:
        print("Nie udało się zaimportować Trainer. Upewnij się, że PYTHONPATH jest poprawnie ustawiony.")
        print("Lub uruchom skrypt z głównego katalogu projektu używając: python -m src.run_grid_search")
        sys.exit(1)

    main()
