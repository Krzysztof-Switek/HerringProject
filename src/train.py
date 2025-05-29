import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from data_loader.dataset import HerringDataset
from models.model import HerringModel
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import csv
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class Trainer:
    def __init__(self, config_path: str = None):
        self.project_root = Path(__file__).parent.parent
        print(f"\nProject root: {self.project_root}")
        self.cfg = self._load_config(config_path)
        self.device = self._init_device()
        print(f"Using device: {self.device}")
        self._validate_data_structure()
        self.model = HerringModel(self.cfg).to(self.device)
        self.data_loader = HerringDataset(self.cfg)

        current_date = datetime.now().strftime("%d-%m")
        model_name = self.cfg.model.base_model
        log_dir = self.project_root / "results" / "logs" / f"{model_name}_{current_date}"
        log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = self._init_tensorboard(log_dir)
        self.log_dir = log_dir
        self.best_acc = 0.0  # ✅ DODANE: dla zapisu najlepszego modelu
        self.early_stop_counter = 0  # ✅ DODANE: licznik do early stopping
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

    def _save_augment_summary(self):
        if hasattr(self.data_loader, 'class_counts'):
            output_path = self.log_dir / "augment_usage_summary.csv"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("Populacja,Wiek,AugmentacjaZastosowana,Łącznie\n")
                for (pop, wiek), total in sorted(self.data_loader.class_counts.items()):
                    used = self.data_loader.augment_applied.get((pop, wiek), 0) if hasattr(self.data_loader,
                                                                                           'augment_applied') else 0
                    f.write(f"{pop},{wiek},{used},{total}\n")
            print(f"📈 Augmentacja per klasa zapisana do: {output_path}")

    def _save_confusion_matrix(self):
        if self.best_cm is not None and self.class_names:
            cm_path = self.log_dir / "confusion_matrix_best_model.npz"
            np.savez(cm_path, matrix=self.best_cm, labels=np.array(self.class_names))
            print(f"📊 Confusion matrix with labels saved to: {cm_path}")

    def _load_config(self, config_path: str = None):
        if config_path is None:
            config_path = self.project_root / "src" / "config" / "config.yaml"

        cfg = OmegaConf.load(config_path)

        if not Path(cfg.data.metadata_file).is_absolute():
            cfg.data.metadata_file = str(self.project_root / cfg.data.metadata_file)
        if not Path(cfg.data.root_dir).is_absolute():
            cfg.data.root_dir = str(self.project_root / cfg.data.root_dir)
        if not Path(cfg.training.checkpoint_dir).is_absolute():
            cfg.training.checkpoint_dir = str(self.project_root / cfg.training.checkpoint_dir)

        return cfg

    def _init_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _validate_data_structure(self):
        print("\nValidating data structure...")
        for split in ["train", "val"]:
            split_path = Path(self.cfg.data.root_dir) / split
            if not split_path.exists():
                raise FileNotFoundError(f"Missing directory: {split_path}")
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

    def _train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        stats = {'loss': 0.0, 'correct': 0, 'total': 0}
        all_targets, all_preds, all_probs = [], [], []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            stats['loss'] += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            targets_np = targets.cpu().numpy()

            stats['correct'] += (preds == targets_np).sum()
            stats['total'] += targets.size(0)
            all_targets.extend(targets_np)
            all_preds.extend(preds)
            all_probs.extend(probs)

        loss = stats['loss'] / len(train_loader)
        acc = 100. * stats['correct'] / stats['total']
        precision = precision_score(all_targets, all_preds)
        recall = recall_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)
        auc = roc_auc_score(all_targets, all_probs)

        return loss, acc, precision, recall, f1, auc, all_targets

    def _validate(self, val_loader, criterion):
        self.model.eval()
        stats = {'loss': 0.0, 'correct': 0, 'total': 0}
        all_targets, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                stats['loss'] += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
                targets_np = targets.cpu().numpy()

                stats['correct'] += (preds == targets_np).sum()
                stats['total'] += targets.size(0)
                all_targets.extend(targets_np)
                all_preds.extend(preds)
                all_probs.extend(probs)

        loss = stats['loss'] / len(val_loader)
        acc = 100. * stats['correct'] / stats['total']
        precision = precision_score(all_targets, all_preds)
        recall = recall_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)
        auc = roc_auc_score(all_targets, all_probs)

        cm = confusion_matrix(all_targets, all_preds)
        return loss, acc, precision, recall, f1, auc, cm, all_targets

    def train(self):
        train_loader, val_loader, class_names = self.data_loader.get_loaders()
        self.class_names = class_names

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.training.learning_rate,
                                weight_decay=self.cfg.model.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.training.epochs)

        model_name = self.cfg.model.base_model
        checkpoint_dir = self.project_root / "checkpoints" / f"{model_name}_{datetime.now().strftime('%d-%m')}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.cfg.training.epochs):
            start_time = time.time()
            train_loss, train_acc, train_prec, train_rec, train_f1, train_auc, train_targets = self._train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_cm, val_targets = self._validate(val_loader, criterion)
            epoch_time = time.time() - start_time

            train_c0, train_c1 = self._get_class_distribution(train_targets)
            val_samples = len(val_targets)
            train_samples = len(train_targets)

            print(f"\nEpoch {epoch + 1}/{self.cfg.training.epochs}:")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Precision: {train_prec:.2f}, Recall: {train_rec:.2f}, F1: {train_f1:.2f}, AUC: {train_auc:.2f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Precision: {val_prec:.2f}, Recall: {val_rec:.2f}, F1: {val_f1:.2f}, AUC: {val_auc:.2f}")
            print(f"Train class dist: 0: {train_c0}, 1: {train_c1}, Time: {epoch_time:.1f}s")

            self.metrics_writer.writerow([
                epoch + 1, train_samples, val_samples, train_c0, train_c1,
                train_loss, train_acc, train_prec, train_rec, train_f1, train_auc,
                val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, round(epoch_time, 2)
            ])

            self._save_augment_summary()

            # ✅ DODANE: zapis najlepszego modelu na podstawie ACC
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_cm = val_cm
                model_path = checkpoint_dir / f"{model_name}_ACC_{val_acc:.2f}.pth"
                torch.save(self.model.state_dict(), model_path)
                print(f"💾 Zapisano najlepszy model do: {model_path}")
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                print(f"⚠️ Early stop counter: {self.early_stop_counter}")

            # ✅ DODANE: warunek early stopping
            if self.early_stop_counter >= self.cfg.training.early_stopping_patience:
                print(f"🛑 Trening przerwany po {epoch + 1} epokach z powodu braku poprawy ACC")
                break

            scheduler.step()

            if getattr(self.cfg.training, "stop_after_one_epoch", False):
                assert self.metrics_file.tell() > 0, "❌ training_metrics.csv jest pusty!"
                assert (self.log_dir / "augment_usage_summary.csv").exists(), "❌ augment_usage_summary.csv nie został zapisany!"
                print("🛑 Trening przerwany po jednej epoce – tryb testowy pipeline'u.")
                break

        self.metrics_file.close()
        if self.writer:
            self.writer.close()

        self._save_confusion_matrix()
        self._save_augment_summary()

if __name__ == "__main__":
    try:
        trainer = Trainer()
        trainer.train()
    except Exception as e:
        print(f"Error: {e}")
        raise
##@