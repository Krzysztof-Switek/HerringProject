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


# ✅ DODANE: definicja ważonych wag dla klas wiekowych w populacji 2
def get_sample_weights(populacje, wieki):
    weight_map = {
        (2, 3): 1.3,
        (2, 4): 1.5,
        (2, 5): 1.6,
        (2, 6): 1.8,
        (2, 7): 2.0,
        (2, 8): 2.0,
        (2, 9): 2.0,
        (2, 10): 2.0
    }
    weights = [weight_map.get((int(p), int(w)), 1.0) for p, w in zip(populacje, wieki)]
    return torch.tensor(weights, dtype=torch.float32)


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

        for inputs, targets, meta in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            weights = get_sample_weights(meta['populacja'], meta['wiek']).to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets, reduction='none')
            loss = (loss * weights).mean()
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
            for inputs, targets, meta in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                weights = get_sample_weights(meta['populacja'], meta['wiek']).to(self.device)
                outputs = self.model(inputs)
                loss = nn.functional.cross_entropy(outputs, targets, reduction='none')
                loss = (loss * weights).mean()
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

        criterion = nn.CrossEntropyLoss(reduction='none')
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

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_cm = val_cm
                model_path = checkpoint_dir / f"{model_name}_ACC_{val_acc:.2f}.pth"
                torch.save(self.model.state_dict(), model_path)
                print(f"\U0001f4be Zapisano najlepszy model do: {model_path}")
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                print(f"\u26a0\ufe0f Early stop counter: {self.early_stop_counter}")

            if self.early_stop_counter >= self.cfg.training.early_stopping_patience:
                print(f"\U0001f6d1 Trening przerwany po {epoch + 1} epokach z powodu braku poprawy ACC")
                break

            scheduler.step()

            if getattr(self.cfg.training, "stop_after_one_epoch", False):
                assert self.metrics_file.tell() > 0, "\u274c training_metrics.csv jest pusty!"
                assert (self.log_dir / "augment_usage_summary.csv").exists(), "\u274c augment_usage_summary.csv nie został zapisany!"
                print("\U0001f6d1 Trening przerwany po jednej epoce – tryb testowy pipeline'u.")
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

    #