import time
from datetime import datetime
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from data_loader.dataset import HerringDataset
from models.model import build_model
from models.model_config import MODEL_CONFIGS
from engine.loss_utills import LossFactory, MultiTaskLossWrapper
from .trainer_metadata import get_class_metadata
from .trainer_logger import (
    init_metrics_logger,
    log_epoch_metrics,
    save_best_model,
    should_stop_early
)
from engine.train_loop import train_epoch, validate
from engine.predict_after_training import run_full_dataset_prediction

from pathlib import Path


def run_training_loop(trainer):
    """Główna pętla treningowa. Zwraca krotkę (best_score, log_dir) dla Optuny."""
    # Load model-specific config and merge it into the main config
    model_name = trainer.cfg.model_name
    if model_name in MODEL_CONFIGS:
        model_cfg = OmegaConf.create(MODEL_CONFIGS[model_name])
        trainer.cfg = OmegaConf.merge(trainer.cfg, model_cfg)
    else:
        raise ValueError(f"Model '{model_name}' not found in MODEL_CONFIGS.")

    trainer.data_loader = HerringDataset(trainer.cfg, path_manager=trainer.path_manager, population_mapper=trainer.population_mapper)
    train_loader, val_loader, class_names = trainer.data_loader.get_loaders()
    trainer.class_names = class_names

    is_multitask = trainer.cfg.mode == 'multitask'

    checkpoint_root = trainer.path_manager.checkpoint_dir()
    logs_root = trainer.path_manager.logs_dir()
    metadata = get_class_metadata(trainer)
    class_counts = metadata["class_counts"]
    class_freq = metadata["class_freq"]

    for loss_name in trainer.cfg.training.loss_type:
        print(f"\n🎯 Start treningu z funkcją straty: {loss_name}")

        loss_factory = LossFactory(
            loss_name,
            class_counts=class_counts if loss_name in ["ldam", "seesaw"] else None,
            class_freq=class_freq if loss_name == "class_balanced_focal" else None
        )
        classification_loss = loss_factory.get()

        # Loss wrapper
        if is_multitask:
            loss_fn = MultiTaskLossWrapper(
                classification_loss=classification_loss,
                regression_loss=torch.nn.MSELoss(),
                method=trainer.cfg.multitask_model.loss_weighting.method,
                static_weights=getattr(trainer.cfg.multitask_model.loss_weighting, "static", None)
            )
        else:
            loss_fn = classification_loss

        # Model
        trainer.model = build_model(trainer.cfg).to(trainer.device)

        optimizer = optim.AdamW(
            trainer.model.parameters(),
            lr=trainer.cfg.training.learning_rate,
            weight_decay=trainer.cfg.base_model.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=trainer.cfg.training.epochs
        )

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        if is_multitask:
            mode_type = "multi"
        elif getattr(trainer.cfg, "expert_model", {}).get("use", False):
            mode_type = "expert"
        else:
            mode_type = "basic"
        full_name = f"{model_name}_{loss_name}_{mode_type}_{timestamp}"
        log_dir = logs_root / full_name
        checkpoint_dir = checkpoint_root / full_name
        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        trainer.writer = SummaryWriter(log_dir=str(log_dir))
        trainer.log_dir = log_dir
        # trainer.best_acc = 0.0 # Zastąpione przez best_score
        trainer.best_score = -float('inf')  # Inicjalizacja dla composite_score
        trainer.early_stop_counter = 0
        trainer.best_cm = None  # Pozostaje, jeśli chcemy logować CM najlepszego modelu

        init_metrics_logger(trainer, log_dir, full_name)

        # Zapis minimalistycznej konfiguracji przebiegu do params.yaml
        params_to_save = {
            'model_name_used': model_name,  # Używamy zmiennej model_name zdefiniowanej wcześniej w pętli
            'model_mode': mode_type,  # Używamy zmiennej mode_type zdefiniowanej wcześniej w pętli
            'loss_function_used': loss_name,  # Bieżąca funkcja straty z pętli
            'run_timestamp': timestamp  # Timestamp dla tego konkretnego przebiegu (loss_name + model_mode)
        }

        # Dodaj wagi composite_score, jeśli model jest multitask i wagi są zdefiniowane
        if mode_type == "multitask" and \
                hasattr(trainer.cfg, 'multitask_model') and \
                trainer.cfg.multitask_model and \
                hasattr(trainer.cfg.multitask_model, 'metrics_weights') and \
                trainer.cfg.multitask_model.metrics_weights:
            try:
                weights = trainer.cfg.multitask_model.metrics_weights
                params_to_save['composite_score_weights'] = {
                    'alpha': weights.alpha,
                    'beta': weights.beta,
                    'gamma': weights.gamma
                }
            except Exception as e:
                print(f"⚠️ Nie udało się pobrać wag composite_score do params.yaml: {e}")

        params_file_path = log_dir / "params.yaml"
        try:
            omega_conf_to_save = OmegaConf.create(params_to_save)
            OmegaConf.save(config=omega_conf_to_save, f=str(params_file_path))
            print(
                f"💾 Zapisano minimalistyczną konfigurację przebiegu (model, tryb, loss, timestamp, wagi) do: {params_file_path}")
            if trainer.debug_mode:  # Użyj trainer.debug_mode
                print(f"   Zawartość params.yaml dla debugu: {OmegaConf.to_yaml(omega_conf_to_save)}")
        except Exception as e:
            print(f"⚠️ Nie udało się zapisać minimalistycznej konfiguracji przebiegu (params.yaml): {e}")

        for epoch in range(trainer.cfg.training.epochs):
            start_time = time.time()

            train_metrics = train_epoch(
                model=trainer.model,
                device=trainer.device,
                dataloader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                population_mapper=trainer.population_mapper
            )

            val_metrics = validate(
                model=trainer.model,
                device=trainer.device,
                dataloader=val_loader,
                loss_fn=loss_fn,
                population_mapper=trainer.population_mapper,
                cfg=trainer.cfg  # <-- Przekazanie konfiguracji
            )

            epoch_time = time.time() - start_time

            log_epoch_metrics(
                trainer,
                epoch,
                loss_name,
                train_metrics,
                val_metrics,
                epoch_time
            )

            # Pobieramy composite_score z metryk walidacyjnych
            current_composite_score = val_metrics.get("composite_score", np.nan)
            # Jeśli composite_score to NaN, save_best_model sobie z tym poradzi

            trainer, improved = save_best_model(
                trainer,
                current_composite_score,  # Przekazujemy composite_score
                val_metrics["cm"],  # CM nadal jest przydatne
                model_name,
                loss_name,
                checkpoint_dir
            )

            if not improved:
                trainer.early_stop_counter += 1
                print(f"⚠️ Early stop counter: {trainer.early_stop_counter}")

            if should_stop_early(trainer):
                print(f"🚍 Trening ({loss_name}) przerwany po {epoch + 1} epokach z powodu braku poprawy ACC")
                break

            scheduler.step()

            if getattr(trainer.cfg.training, "stop_after_one_epoch", False):
                print("🛑 Trening przerwany po jednej epoce – tryb testowy pipeline'u.")
                break

        trainer.metrics_file.close()
        if trainer.writer:
            trainer.writer.close()

        if trainer.last_model_path is not None:
            print(f"🔍 Uruchamianie predykcji dla {loss_name} na całym zbiorze...")
            from .trainer_logger import log_augmentation_summary
            if hasattr(trainer.data_loader, 'augment_applied') and isinstance(trainer.data_loader.augment_applied,
                                                                              dict):
                log_augmentation_summary(trainer.data_loader.augment_applied, full_name, log_dir=log_dir)
            else:
                print(
                    "⚠️ Dane o augmentacji nie są dostępne lub są w nieprawidłowym formacie, pomijam logowanie podsumowania augmentacji.")

            limit_for_prediction = 100 if trainer.debug_mode else None
            run_full_dataset_prediction(
                loss_name=loss_name,
                model_path=str(trainer.last_model_path),
                path_manager=trainer.path_manager,
                log_dir=log_dir,
                full_name=full_name,
                limit_predictions=limit_for_prediction
            )

            try:
                from utils.Training_Prediction_Report import TrainingPredictionReport
                predictions_path = log_dir / f"{full_name}_predictions.xlsx"
                metrics_file_path = log_dir / f"{full_name}_training_metrics.csv"
                augmentation_file_path = log_dir / f"augmentation_summary_{full_name}.csv"

                run_params_data = None
                if params_file_path.exists():
                    try:
                        run_params_data = OmegaConf.load(params_file_path)
                    except Exception as e_params:
                        print(f"⚠️ Nie udało się wczytać params.yaml ({params_file_path}) dla raportu: {e_params}")

                print(f"📑 Generuję raport PDF podsumowujący cały trening oraz predykcję w: {log_dir}")
                report = TrainingPredictionReport(
                    log_dir=log_dir,
                    base_config_obj=trainer.cfg,
                    run_params_obj=run_params_data,
                    predictions_path=predictions_path,
                    metadata_path=trainer.path_manager.metadata_file(),
                    metrics_path=metrics_file_path if metrics_file_path.exists() else None,
                    augmentation_path=augmentation_file_path if augmentation_file_path.exists() else None
                )
                report.run()
            except Exception as e:
                print(f"⚠️ Nie udało się wygenerować raportu PDF: {e}")

        else:
            print(f"⚠️ Brak zapisanego modelu dla {loss_name}, predykcja pominięta.")

    # --- NOWA SEKCJA ZWRACAJĄCA WYNIK DLA OPTUNY ---
    best_score_for_this_run = trainer.best_score
    if not np.isfinite(best_score_for_this_run):
        best_score_for_this_run = 0.0

    return best_score_for_this_run, trainer.log_dir
