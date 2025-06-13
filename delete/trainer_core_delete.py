import time
import torch
import pandas as pd
import numpy as np
import csv
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from engine.train_loop import train_epoch, validate
from engine.delete.loss_utils_old_delete import LossFactory
from models.model import HerringModel
from engine.predict_after_training import run_full_dataset_prediction
import torch.optim as optim


def run_training_loop(trainer):
    train_loader, val_loader, class_names = trainer.data_loader.get_loaders()
    trainer.class_names = class_names

    model_name = trainer.cfg.model.base_model
    checkpoint_root = trainer.path_manager.checkpoint_dir()
    logs_root = trainer.path_manager.logs_dir()

    df = pd.read_excel(trainer.path_manager.metadata_file())
    df_train = df[df["SET"].str.lower() == "train"]
    age_counts = df_train["Wiek"].value_counts().sort_index().to_dict()
    class_counts = [age_counts.get(age, 0) for age in sorted(age_counts)]

    for loss_name in trainer.cfg.training.loss_type:
        print(f"\n🎯 Start treningu z funkcją straty: {loss_name}")

        loss_factory = LossFactory(loss_name, class_counts=class_counts if loss_name == "ldam" else None)
        criterion = loss_factory.get()

        trainer.model = HerringModel(trainer.cfg).to(trainer.device)
        optimizer = optim.AdamW(trainer.model.parameters(), lr=trainer.cfg.training.learning_rate,
                                weight_decay=trainer.cfg.model.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trainer.cfg.training.epochs)

        timestamp = datetime.now().strftime('%d-%m_%H-%M')
        log_dir = logs_root / f"{model_name}_{loss_name}_{timestamp}"
        checkpoint_dir = checkpoint_root / f"{model_name}_{loss_name}_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        trainer.writer = SummaryWriter(log_dir=str(log_dir))
        trainer.log_dir = log_dir
        trainer.best_acc = 0.0
        trainer.early_stop_counter = 0
        trainer.best_cm = None

        metrics_file_path = log_dir / "training_metrics.csv"
        trainer.metrics_file = open(metrics_file_path, mode="w", newline="")
        trainer.metrics_writer = csv.writer(trainer.metrics_file)
        trainer.metrics_writer.writerow([
            'Epoch', 'Train Samples', 'Val Samples', 'Train Class 0', 'Train Class 1',
            'Train Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Train AUC',
            'Val Loss', 'Val Accuracy', 'Val Precision', 'Val Recall', 'Val F1', 'Val AUC',
            'Train Time (s)'
        ])

        def get_class_distribution(targets):
            values, counts = np.unique(targets, return_counts=True)
            class_dist = {int(v): int(c) for v, c in zip(values, counts)}
            return class_dist.get(0, 0), class_dist.get(1, 0)

        for epoch in range(trainer.cfg.training.epochs):
            start_time = time.time()
            train_loss, train_acc, train_prec, train_rec, train_f1, train_auc, train_targets = train_epoch(
                trainer.model, trainer.device, train_loader, optimizer, criterion
            )
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_cm, val_targets = validate(
                trainer.model, trainer.device, val_loader, criterion
            )
            epoch_time = time.time() - start_time

            train_c0, train_c1 = get_class_distribution(train_targets)
            val_samples = len(val_targets)
            train_samples = len(train_targets)

            print(f"\nEpoch {epoch + 1}/{trainer.cfg.training.epochs} ({loss_name}):")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Precision: {train_prec:.2f}, Recall: {train_rec:.2f}, F1: {train_f1:.2f}, AUC: {train_auc:.2f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Precision: {val_prec:.2f}, Recall: {val_rec:.2f}, F1: {val_f1:.2f}, AUC: {val_auc:.2f}")
            print(f"Train class dist: 0: {train_c0}, 1: {train_c1}, Time: {epoch_time:.1f}s")

            trainer.metrics_writer.writerow([
                f"{loss_name}-e{epoch + 1}", train_samples, val_samples, train_c0, train_c1,
                train_loss, train_acc, train_prec, train_rec, train_f1, train_auc,
                val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, round(epoch_time, 2)
            ])

            if val_acc > trainer.best_acc:
                trainer.best_acc = val_acc
                trainer.best_cm = val_cm
                model_path = checkpoint_dir / f"{model_name}_{loss_name}_ACC_{val_acc:.2f}.pth"
                torch.save(trainer.model.state_dict(), model_path)
                trainer.last_model_path = model_path
                print(f"💾 Zapisano najlepszy model do: {model_path}")
                trainer.early_stop_counter = 0
            else:
                trainer.early_stop_counter += 1
                print(f"⚠️ Early stop counter: {trainer.early_stop_counter}")

            if trainer.early_stop_counter >= trainer.cfg.training.early_stopping_patience:
                print(f"🚝 Trening ({loss_name}) przerwany po {epoch + 1} epokach z powodu braku poprawy ACC")
                break

            scheduler.step()

            if getattr(trainer.cfg.training, "stop_after_one_epoch", False):
                print("🚝 Trening przerwany po jednej epoce – tryb testowy pipeline'u.")
                break

        trainer.metrics_file.close()
        if trainer.writer:
            trainer.writer.close()

        if trainer.last_model_path is not None:
            print(f"🔍 Uruchamianie predykcji dla {loss_name} na całym zbiorze...")
            run_full_dataset_prediction(
                loss_name=loss_name,
                model_path=str(trainer.last_model_path),
                path_manager=trainer.path_manager
            )
        else:
            print(f"⚠️ Brak zapisanego modelu dla {loss_name}, predykcja pominięta.")
