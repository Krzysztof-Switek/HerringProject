import torch
import torch.optim as optim
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from data_loader.dataset import HerringDataset
from models.model import HerringModel
import csv
import time
import numpy as np
from datetime import datetime
from engine.train_loop import train_epoch, validate
from engine.loss_utils import LossFactory
from engine.predict_after_training import run_full_dataset_prediction
import pandas as pd
from utils.path_manager import PathManager


class Trainer:
    def __init__(self, config_path: str = None, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        print(f"\nProject root: {self.project_root}")
        self.cfg = self._load_config(config_path)
        self.path_manager = PathManager(self.project_root, self.cfg)
        self.device = self._init_device()
        print(f"Using device: {self.device}")
        self._validate_data_structure()
        self.model = None
        self.data_loader = HerringDataset(self.cfg)
        self.last_model_path = None  # 🟢 do przechowania najlepszego modelu

        current_date = datetime.now().strftime("%d-%m")
        model_name = self.cfg.model.base_model
        log_dir = self.path_manager.logs_dir() / f"{model_name}_{current_date}"
        log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = self._init_tensorboard(log_dir)
        self.log_dir = log_dir
        self.best_acc = 0.0
        self.early_stop_counter = 0
        self.best_cm = None
        self.class_names = []

        metrics_file_path = log_dir / "training_metrics.csv"
        self.metrics_file = open(metrics_file_path, mode="w", newline="")
        self.metrics_writer = csv.writer(self.metrics_file)
        self.metrics_writer.writerow([
            'Epoch', 'Train Samples', 'Val Samples', 'Train Class 0', 'Train Class 1',
            'Train Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Train AUC',
            'Val Loss', 'Val Accuracy', 'Val Precision', 'Val Recall', 'Val F1', 'Val AUC',
            'Train Time (s)'
        ])

    def _load_config(self, config_path: str = None):
        if config_path is None:
            temp_path_manager = PathManager(self.project_root, cfg=None)
            config_path = temp_path_manager.config_path()
        return OmegaConf.load(config_path)

    def _init_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _validate_data_structure(self):
        print("\nValidating data structure...")
        for split in ["train", "val"]:
            split_path = self.path_manager.data_root() / split
            if not split_path.exists():
                raise FileNotFoundError(f"Missing directory: {split_path}")
            if not any(split_path.iterdir()):
                raise RuntimeError(f"Katalog {split_path} istnieje, ale jest pusty")
        print("Data structure validated.")

    def _init_tensorboard(self, log_dir):
        try:
            return SummaryWriter(log_dir=str(log_dir))
        except Exception as e:
            print(f"TensorBoard init failed: {e}")
            return None

    def _get_class_distribution(self, targets):
        values, counts = np.unique(targets, return_counts=True)
        class_dist = {int(v): int(c) for v, c in zip(values, counts)}
        return class_dist.get(0, 0), class_dist.get(1, 0)

    def train(self):
        train_loader, val_loader, class_names = self.data_loader.get_loaders()
        self.class_names = class_names

        model_name = self.cfg.model.base_model
        checkpoint_root = self.path_manager.checkpoint_dir()
        logs_root = self.path_manager.logs_dir()

        df = pd.read_excel(self.path_manager.metadata_file())
        df_train = df[df["SET"].str.lower() == "train"]
        age_counts = df_train["Wiek"].value_counts().sort_index().to_dict()
        class_counts = [age_counts.get(age, 0) for age in sorted(age_counts)]

        for loss_name in self.cfg.training.loss_type:
            print(f"\n🎯 Start treningu z funkcją straty: {loss_name}")

            loss_factory = LossFactory(loss_name, class_counts=class_counts if loss_name == "ldam" else None)
            criterion = loss_factory.get()

            self.model = HerringModel(self.cfg).to(self.device)
            optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.training.learning_rate,
                                    weight_decay=self.cfg.model.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.training.epochs)

            timestamp = datetime.now().strftime('%d-%m_%H-%M')
            log_dir = logs_root / f"{model_name}_{loss_name}_{timestamp}"
            checkpoint_dir = checkpoint_root / f"{model_name}_{loss_name}_{timestamp}"
            log_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            self.writer = self._init_tensorboard(log_dir)
            self.log_dir = log_dir
            self.best_acc = 0.0
            self.early_stop_counter = 0
            self.best_cm = None

            metrics_file_path = log_dir / "training_metrics.csv"
            self.metrics_file = open(metrics_file_path, mode="w", newline="")
            self.metrics_writer = csv.writer(self.metrics_file)
            self.metrics_writer.writerow([
                'Epoch', 'Train Samples', 'Val Samples', 'Train Class 0', 'Train Class 1',
                'Train Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Train AUC',
                'Val Loss', 'Val Accuracy', 'Val Precision', 'Val Recall', 'Val F1', 'Val AUC',
                'Train Time (s)'
            ])

            for epoch in range(self.cfg.training.epochs):
                start_time = time.time()
                train_loss, train_acc, train_prec, train_rec, train_f1, train_auc, train_targets = train_epoch(
                    self.model, self.device, train_loader, optimizer, criterion
                )
                val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_cm, val_targets = validate(
                    self.model, self.device, val_loader, criterion
                )
                epoch_time = time.time() - start_time

                train_c0, train_c1 = self._get_class_distribution(train_targets)
                val_samples = len(val_targets)
                train_samples = len(train_targets)

                print(f"\nEpoch {epoch + 1}/{self.cfg.training.epochs} ({loss_name}):")
                print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Precision: {train_prec:.2f}, Recall: {train_rec:.2f}, F1: {train_f1:.2f}, AUC: {train_auc:.2f}")
                print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Precision: {val_prec:.2f}, Recall: {val_rec:.2f}, F1: {val_f1:.2f}, AUC: {val_auc:.2f}")
                print(f"Train class dist: 0: {train_c0}, 1: {train_c1}, Time: {epoch_time:.1f}s")

                self.metrics_writer.writerow([
                    f"{loss_name}-e{epoch + 1}", train_samples, val_samples, train_c0, train_c1,
                    train_loss, train_acc, train_prec, train_rec, train_f1, train_auc,
                    val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, round(epoch_time, 2)
                ])

                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self.best_cm = val_cm
                    model_path = checkpoint_dir / f"{model_name}_{loss_name}_ACC_{val_acc:.2f}.pth"
                    torch.save(self.model.state_dict(), model_path)
                    self.last_model_path = model_path  # 🟢 zapamiętajmy do predykcji
                    print(f"💾 Zapisano najlepszy model do: {model_path}")
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    print(f"⚠️ Early stop counter: {self.early_stop_counter}")

                if self.early_stop_counter >= self.cfg.training.early_stopping_patience:
                    print(f"🚝 Trening ({loss_name}) przerwany po {epoch + 1} epokach z powodu braku poprawy ACC")
                    break

                scheduler.step()

                if getattr(self.cfg.training, "stop_after_one_epoch", False):
                    print("🚝 Trening przerwany po jednej epoce – tryb testowy pipeline'u.")
                    break

            self.metrics_file.close()
            if self.writer:
                self.writer.close()

            # 🟢 Predykcja po treningu (jeśli zapisano model)
            if self.last_model_path is not None:
                print(f"🔍 Uruchamianie predykcji dla {loss_name} na całym zbiorze...")
                run_full_dataset_prediction(
                    loss_name=loss_name,
                    model_path=str(self.last_model_path),
                    path_manager=self.path_manager
                )
            else:
                print(f"⚠️ Brak zapisanego modelu dla {loss_name}, predykcja pominięta.")
