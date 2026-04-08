from __future__ import annotations

import time
from datetime import datetime

import torch
import torch.optim as optim


class _NoOpSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def add_scalars(self, *args, **kwargs):
        return None

    def add_histogram(self, *args, **kwargs):
        return None

    def add_figure(self, *args, **kwargs):
        return None

    def add_text(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


def _create_summary_writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=str(log_dir))
    except Exception:
        print("⚠️ TensorBoard niedostępny - używam no-op SummaryWriter.")
        return _NoOpSummaryWriter()


def _import_loss_components():
    try:
        from .loss_utills import LossFactory, MultiTaskLossWrapper
        return LossFactory, MultiTaskLossWrapper
    except ImportError:
        from src.engine.loss_utills import LossFactory, MultiTaskLossWrapper
        return LossFactory, MultiTaskLossWrapper


def _import_train_loop():
    try:
        from .train_loop import train_epoch, validate
        return train_epoch, validate
    except ImportError:
        from src.engine.train_loop import train_epoch, validate
        return train_epoch, validate


def _import_prediction_runner():
    try:
        from .predict_after_training import run_full_dataset_prediction
        return run_full_dataset_prediction
    except ImportError:
        from src.engine.predict_after_training import run_full_dataset_prediction
        return run_full_dataset_prediction


def _import_model_builder():
    try:
        from ..models.model import build_model
        return build_model
    except ImportError:
        from src.models.model import build_model
        return build_model


def _import_trainer_metadata():
    try:
        from .trainer_metadata import get_class_metadata
        return get_class_metadata
    except ImportError:
        from src.engine.trainer_metadata import get_class_metadata
        return get_class_metadata


def _import_trainer_logger():
    try:
        from .trainer_logger import (
            init_metrics_logger,
            log_epoch_metrics,
            save_best_model,
            should_stop_early,
            log_augmentation_summary,
        )
        return (
            init_metrics_logger,
            log_epoch_metrics,
            save_best_model,
            should_stop_early,
            log_augmentation_summary,
        )
    except ImportError:
        from src.engine.trainer_logger import (
            init_metrics_logger,
            log_epoch_metrics,
            save_best_model,
            should_stop_early,
            log_augmentation_summary,
        )
        return (
            init_metrics_logger,
            log_epoch_metrics,
            save_best_model,
            should_stop_early,
            log_augmentation_summary,
        )


def _import_training_prediction_report():
    try:
        from ..utils.Training_Prediction_Report import TrainingPredictionReport
        return TrainingPredictionReport
    except ImportError:
        from src.utils.Training_Prediction_Report import TrainingPredictionReport
        return TrainingPredictionReport


def run_training_loop(trainer):
    LossFactory, MultiTaskLossWrapper = _import_loss_components()
    train_epoch, validate = _import_train_loop()
    run_full_dataset_prediction = _import_prediction_runner()
    build_model = _import_model_builder()
    get_class_metadata = _import_trainer_metadata()
    (
        init_metrics_logger,
        log_epoch_metrics,
        save_best_model,
        should_stop_early,
        log_augmentation_summary,
    ) = _import_trainer_logger()

    train_loader, val_loader, class_names = trainer.data_loader.get_loaders()
    trainer.class_names = class_names

    is_multitask = bool(trainer.cfg.multitask_model.use)
    model_name = (
        trainer.cfg.multitask_model.backbone_model.model_name
        if is_multitask
        else trainer.cfg.base_model.base_model
    )

    checkpoint_root = trainer.path_manager.checkpoint_dir()
    logs_root = trainer.path_manager.logs_dir()
    metadata = get_class_metadata(trainer)
    class_counts = metadata["class_counts"]
    class_freq = metadata["class_freq"]

    # --- BUG 1 FIX ---
    # weight_decay musi pochodzić z sekcji odpowiadającej aktywnemu modelowi.
    # W trybie multitask parametry modelu są definiowane w multitask_model.backbone_model,
    # a nie w base_model — użycie base_model.weight_decay w trybie multitask
    # powoduje trenowanie z błędną regularyzacją bez żadnego ostrzeżenia.
    weight_decay = (
        trainer.cfg.multitask_model.backbone_model.weight_decay
        if is_multitask
        else trainer.cfg.base_model.weight_decay
    )

    for loss_name in trainer.cfg.training.loss_type:
        print(f"\n🎯 Start treningu z funkcją straty: {loss_name}")

        loss_factory = LossFactory(
            loss_name,
            class_counts=class_counts if loss_name in ["ldam", "seesaw"] else None,
            class_freq=class_freq if loss_name == "class_balanced_focal" else None,
        )
        classification_loss = loss_factory.get()

        if hasattr(classification_loss, 'precompute_from_counts'):
            classification_loss.precompute_from_counts(trainer.data_loader.class_counts)

        if is_multitask:
            loss_fn = MultiTaskLossWrapper(
                classification_loss=classification_loss,
                regression_loss=torch.nn.MSELoss(),
                method=trainer.cfg.multitask_model.loss_weighting.method,
                static_weights=getattr(
                    trainer.cfg.multitask_model.loss_weighting,
                    "static",
                    None,
                ),
            )
        else:
            loss_fn = classification_loss

        trainer.model = build_model(trainer.cfg).to(trainer.device)

        # --- BUG 2 FIX ---
        # MultiTaskLossWrapper przy method="uncertainty" posiada self.log_vars = nn.Parameter,
        # a przy method="gradnorm" posiada self.weights = nn.Parameter.
        # Bez dodania tych parametrów do optymalizatora są one zamrożone przez cały trening
        # — uncertainty weighting i gradnorm nie działają zgodnie z założeniem.
        optimizer_params = list(trainer.model.parameters())
        if is_multitask:
            learnable_loss_params = list(loss_fn.parameters())
            if learnable_loss_params:
                optimizer_params += learnable_loss_params

        optimizer = optim.AdamW(
            optimizer_params,
            lr=trainer.cfg.training.learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=trainer.cfg.training.epochs,
        )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
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

        trainer.writer = _create_summary_writer(log_dir)
        trainer.log_dir = log_dir
        trainer.best_acc = 0.0
        trainer.early_stop_counter = 0
        trainer.best_cm = None
        trainer.data_loader.augment_applied.clear()  # reset per loss_type
        # --- BUG 3 FIX ---
        # last_model_path jest inicjalizowane w Trainer.__init__, ale nie jest resetowane
        # między kolejnymi funkcjami strat w tej pętli. Gdyby loss_N zapisał model,
        # a loss_N+1 nie poprawił ACC (save_best_model nigdy nie aktualizuje
        # last_model_path), predykcja loss_N+1 uruchomiłaby się na modelu loss_N.
        trainer.last_model_path = None

        init_metrics_logger(trainer, log_dir, full_name)

        for epoch in range(trainer.cfg.training.epochs):
            start_time = time.time()

            train_metrics = train_epoch(
                model=trainer.model,
                device=trainer.device,
                dataloader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                population_mapper=trainer.population_mapper,
            )

            val_metrics = validate(
                model=trainer.model,
                device=trainer.device,
                dataloader=val_loader,
                loss_fn=loss_fn,
                population_mapper=trainer.population_mapper,
            )

            epoch_time = time.time() - start_time

            log_epoch_metrics(
                trainer,
                epoch,
                loss_name,
                train_metrics,
                val_metrics,
                epoch_time,
            )

            val_acc = val_metrics["acc"]
            trainer, improved = save_best_model(
                trainer,
                val_acc,
                val_metrics["cm"],
                model_name,
                loss_name,
                checkpoint_dir,
            )

            if not improved:
                trainer.early_stop_counter += 1
                print(f"⚠️ Early stop counter: {trainer.early_stop_counter}")

            if should_stop_early(trainer):
                print(
                    f"🚍 Trening ({loss_name}) przerwany po {epoch + 1} epokach "
                    f"z powodu braku poprawy ACC"
                )
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

            log_augmentation_summary(
                trainer.data_loader.augment_applied,
                full_name,
                log_dir=log_dir,
            )

            run_full_dataset_prediction(
                loss_name=loss_name,
                model_path=str(trainer.last_model_path),
                path_manager=trainer.path_manager,
                log_dir=log_dir,
                full_name=full_name,
            )

            try:
                TrainingPredictionReport = _import_training_prediction_report()
                predictions_path = log_dir / f"{full_name}_predictions.xlsx"
                print(
                    f"📑 Generuję raport PDF podsumowujący cały trening oraz predykcję w: {log_dir}"
                )
                report = TrainingPredictionReport(
                    log_dir=log_dir,
                    config_path=trainer.path_manager.config_path(),
                    predictions_path=predictions_path,
                    metadata_path=trainer.path_manager.metadata_file(),
                    loss_name=loss_name,
                )
                report.run()
            except Exception as e:
                print(f"⚠️ Nie udało się wygenerować raportu PDF: {e}")

        else:
            print(f"⚠️ Brak zapisanego modelu dla {loss_name}, predykcja pominięta.")