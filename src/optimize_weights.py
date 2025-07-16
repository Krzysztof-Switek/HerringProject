import optuna
from omegaconf import OmegaConf
import sys
from pathlib import Path

# Dodaj ścieżkę do roota projektu, aby umożliwić importy z src
# Zakładamy, że ten skrypt jest w src/
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.engine.trainer import Trainer
from src.utils.path_manager import PathManager


def objective(trial: optuna.Trial):
    """
    Funkcja celu dla Optuny. Wykonuje jeden pełny trening i zwraca metrykę do maksymalizacji.
    """
    # 1. Użyj PathManager do uzyskania ścieżki do konfiguracji, aby zachować spójność
    pm = PathManager()
    config_path = pm.config_path()
    base_cfg = OmegaConf.load(config_path)

    # 2. Zaproponuj wartości wag alpha i beta (gamma będzie obliczona, aby suma była 1)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    beta = trial.suggest_float("beta", 0.0, 1.0 - alpha)
    gamma = 1.0 - alpha - beta

    # 3. Zaktualizuj obiekt konfiguracji nowymi wagami
    OmegaConf.update(base_cfg, "multitask_model.metrics_weights.alpha", alpha, merge=True)
    OmegaConf.update(base_cfg, "multitask_model.metrics_weights.beta", beta, merge=True)
    OmegaConf.update(base_cfg, "multitask_model.metrics_weights.gamma", gamma, merge=True)

    print(f"\n--- Rozpoczynam Trial {trial.number} ---")
    print(f"Sugerowane wagi: alpha={alpha:.4f}, beta={beta:.4f}, gamma={gamma:.4f}")

    try:
        # 4. Uruchom trening z nadpisaną konfiguracją
        trainer = Trainer(config_override=base_cfg)
        best_composite_score = trainer.train()

        # 5. Zapisz dodatkowe metryki w Optunie (opcjonalnie, ale bardzo przydatne)
        if trainer.log_dir:
            trial.set_user_attr("log_dir", str(trainer.log_dir))

        # Można by też odczytać z pliku metryk i zapisać F1, MAE dla najlepszej epoki
        # ale na razie skupmy się na głównym celu.

        print(f"--- Trial {trial.number} zakończony. Wynik (Composite Score): {best_composite_score:.4f} ---")

        # 6. Zwróć metrykę do maksymalizacji
        return best_composite_score

    except Exception as e:
        print(f"Błąd w trakcie Trial {trial.number}: {e}")
        # Zwróć 0.0 lub podnieś błąd, aby Optuna mogła oznaczyć trial jako FAILED
        # Zwrócenie 0.0 jest bezpieczniejsze, jeśli błędy mogą być losowe (np. CUDA out of memory)
        # i nie chcemy przerywać całej optymalizacji.
        return 0.0


if __name__ == "__main__":
    # Stwórz lub załaduj studium Optuny
    # Użycie bazy danych SQLite pozwala na wznawianie przerwanej optymalizacji
    study = optuna.create_study(
        direction="maximize",
        study_name="herring_multitask_weights_optimization",
        storage="sqlite:///optuna_herring_weights.db",  # Zapis postępów do pliku
        load_if_exists=True
    )

    # Uruchom optymalizację
    # Zacznijmy od 25 prób, aby było szybciej niż 50
    study.optimize(objective, n_trials=25)

    print("\n" + "=" * 30)
    print("--- Zakończono optymalizację ---")
    print(f"Liczba zakończonych prób: {len(study.trials)}")
    print(f"Najlepsza wartość (best_value): {study.best_value:.4f}")
    print("Najlepsze parametry (best_params):")
    for key, value in study.best_params.items():
        print(f"    {key}: {value:.4f}")

    # Wyświetl ramkę danych z wynikami
    df = study.trials_dataframe()
    print("\nPełne wyniki:")
    print(df)
    df.to_csv("optuna_results.csv", index=False)
    print("\nWyniki zapisano również do optuna_results.csv")
