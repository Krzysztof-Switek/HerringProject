import optuna
from omegaconf import OmegaConf
from src.engine.trainer import Trainer

def objective(trial: optuna.Trial):
    base_cfg = OmegaConf.load("src/config/config.yaml")

    # Sugerowanie wag (proste podejście)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    beta = trial.suggest_float("beta", 0.0, 1.0 - alpha)
    gamma = 1.0 - alpha - beta

    # Aktualizacja konfiguracji
    OmegaConf.update(base_cfg, "multitask_model.metrics_weights.alpha", alpha, merge=True)
    OmegaConf.update(base_cfg, "multitask_model.metrics_weights.beta", beta, merge=True)
    OmegaConf.update(base_cfg, "multitask_model.metrics_weights.gamma", gamma, merge=True)

    print(f"\n--- Trial {trial.number} | Wagi: a={alpha:.3f}, b={beta:.3f}, g={gamma:.3f} ---")

    # Uruchomienie treningu z nową konfiguracją
    trainer = Trainer(config_override=base_cfg)
    best_composite_score = trainer.train()

    # Zapisanie dodatkowych informacji w Optunie
    trial.set_user_attr("log_dir", str(trainer.log_dir) if trainer.log_dir else "N/A")

    return best_composite_score

if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="herring_weights_optimization",
        storage="sqlite:///optuna_herring.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=50)

    print(f"\n--- Zakończono ---\nNajlepszy wynik: {study.best_value}\nNajlepsze parametry: {study.best_params}")
