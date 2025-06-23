import time
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.model import HerringModel
from models.multitask_model import MultiTaskHerringModel
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

def run_training_loop(trainer):
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
        trainer.best_acc = 0.0
        trainer.early_stop_counter = 0
        trainer.best_cm = None

        init_metrics_logger(trainer, log_dir, full_name)

        for epoch in range(trainer.cfg.training.epochs):
            start_time = time.time()

            train_metrics = train_epoch(
                model=trainer.model,
                device=trainer.device,
                dataloader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer
            )

            val_metrics = validate(
                model=trainer.model,
                device=trainer.device,
                dataloader=val_loader,
                loss_fn=loss_fn
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

            val_acc = val_metrics["acc"]
            trainer, improved = save_best_model(
                trainer,
                val_acc,
                val_metrics["cm"],
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
            log_augmentation_summary(trainer.data_loader.augment_applied, full_name, log_dir=log_dir)
            run_full_dataset_prediction(
                loss_name=loss_name,
                model_path=str(trainer.last_model_path),
                path_manager=trainer.path_manager,
                log_dir=log_dir,
                full_name=full_name
            )
        else:
            print(f"⚠️ Brak zapisanego modelu dla {loss_name}, predykcja pominięta.")

