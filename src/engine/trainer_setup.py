import time
from datetime import datetime
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from pathlib import Path

from models.model import HerringModel
from models.multitask_model import MultiTaskHerringModel
from engine.loss_utills import LossFactory, MultiTaskLossWrapper
from .trainer_metadata import get_class_metadata
from .trainer_logger import (
    init_metrics_logger,
    log_epoch_metrics,
    save_best_model,
    should_stop_early,
    log_augmentation_summary
)
from engine.train_loop import train_epoch, validate
from engine.predict_after_training import run_full_dataset_prediction
from utils.Training_Prediction_Report import TrainingPredictionReport


def run_training_loop(trainer):
    """Główna pętla treningowa. Zwraca krotkę (best_score, log_dir) dla Optuny."""
    train_loader, val_loader, class_names = trainer.data_loader.get_loaders()
    trainer.class_names = class_names

    is_multitask = trainer.cfg.multitask_model.use
    model_name = (
        trainer.cfg.multitask_model.backbone_model.model_name
        if is_multitask else trainer.cfg.base_model.base_model
    )

    checkpoint_root = trainer.path_manager.checkpoint_dir()
    logs_root = trainer.path_manager.logs_dir()
    metadata = get_class_metadata(trainer)
    class_counts = metadata["class_counts"]
    class_freq = metadata["class_freq"]

    # Pętla po funkcjach straty. Dla Optuny, ta lista powinna zawierać tylko jeden element.
    for loss_name in trainer.cfg.training.loss_type:
        print(f"\n🎯 Start treningu z funkcją straty: {loss_name}")

        loss_factory = LossFactory(
            loss_name,
            class_counts=class_counts if loss_name in ["ldam", "seesaw"] else None,
            class_freq=class_freq if loss_name == "class_balanced_focal" else None
        )
        classification_loss = loss_factory.get()

        if is_multitask:
            loss_fn = MultiTaskLossWrapper(
                classification_loss=classification_loss,
                regression_loss=torch.nn.MSELoss(),
                method=trainer.cfg.multitask_model.loss_weighting.method,
                static_weights=getattr(trainer.cfg.multitask_model.loss_weighting, "static", None)
            )
        else:
            loss_fn = classification_loss

        trainer.model = (
            MultiTaskHerringModel(trainer.cfg).to(trainer.device)
            if is_multitask else
            HerringModel(trainer.cfg).to(trainer.device)
        )

        optimizer = optim.AdamW(
            trainer.model.parameters(),
            lr=trainer.cfg.training.learning_rate,
            weight_decay=trainer.cfg.base_model.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=trainer.cfg.training.epochs
        )

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        mode_type = "multi" if is_multitask else "basic"
        full_name = f"{model_name}_{loss_name}_{mode_type}_{timestamp}"
        log_dir = logs_root / full_name
        checkpoint_dir = checkpoint_root / full_name
        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Ustawienie log_dir w obiekcie trainer, aby był dostępny na zewnątrz
        trainer.log_dir = log_dir

        # Inicjalizacja loggerów
        trainer.writer = SummaryWriter(log_dir=str(log_dir)) if trainer.cfg.training.get("use_tensorboard",
                                                                                         False) else None
        trainer.best_score = -float('inf')
        trainer.early_stop_counter = 0
        trainer.best_cm = None
        init_metrics_logger(trainer, log_dir, full_name)

        params_to_save = {
            'model_name_used': model_name,
            'model_mode': mode_type,
            'loss_function_used': loss_name,
            'run_timestamp': timestamp
        }
        if mode_type == "multitask" and hasattr(trainer.cfg.multitask_model, 'metrics_weights'):
            try:
                weights = trainer.cfg.multitask_model.metrics_weights
                params_to_save['composite_score_weights'] = {
                    'alpha': weights.alpha, 'beta': weights.beta, 'gamma': weights.gamma
                }
            except Exception as e:
                print(f"⚠️ Nie udało się pobrać wag composite_score do params.yaml: {e}")

        params_file_path = log_dir / "params.yaml"
        try:
            omega_conf_to_save = OmegaConf.create(params_to_save)
            OmegaConf.save(config=omega_conf_to_save, f=str(params_file_path))
            print(f"💾 Zapisano konfigurację przebiegu do: {params_file_path}")
        except Exception as e:
            print(f"⚠️ Nie udało się zapisać konfiguracji przebiegu (params.yaml): {e}")

        for epoch in range(trainer.cfg.training.epochs):
            start_time = time.time()
            train_metrics = train_epoch(
                trainer.model, trainer.device, train_loader, loss_fn, optimizer, trainer.population_mapper
            )
            val_metrics = validate(
                trainer.model, trainer.device, val_loader, loss_fn, trainer.population_mapper, trainer.cfg
            )
            epoch_time = time.time() - start_time
            log_epoch_metrics(trainer, epoch, loss_name, train_metrics, val_metrics, epoch_time)

            current_composite_score = val_metrics.get("composite_score", np.nan)
            trainer, improved = save_best_model(
                trainer, current_composite_score, val_metrics["cm"], model_name, loss_name, checkpoint_dir
            )

            if not improved:
                trainer.early_stop_counter += 1
            if should_stop_early(trainer):
                print(f"🚍 Trening ({loss_name}) przerwany po {epoch + 1} epokach.")
                break
            scheduler.step()
            if getattr(trainer.cfg.training, "stop_after_one_epoch", False):
                print("🛑 Trening przerwany po jednej epoce (tryb testowy).")
                break

        if hasattr(trainer, 'metrics_file') and trainer.metrics_file:
            trainer.metrics_file.close()
        if trainer.writer:
            trainer.writer.close()

        if trainer.last_model_path is not None:
            if hasattr(trainer.data_loader, 'augment_applied'):
                log_augmentation_summary(trainer.data_loader.augment_applied, full_name, log_dir=log_dir)

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