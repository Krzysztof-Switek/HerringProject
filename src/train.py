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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


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

        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.log_dir = log_dir

        self.csv_path = log_dir / "training_metrics.csv"
        self.metrics_file = open(self.csv_path, mode="w", newline="")
        self.metrics_writer = csv.writer(self.metrics_file)
        self.metrics_writer.writerow([
            'Epoch', 'Train Samples', 'Val Samples', 'Train Class 0', 'Train Class 1',
            'Train Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Train AUC',
            'Val Loss', 'Val Accuracy', 'Val Precision', 'Val Recall', 'Val F1', 'Val AUC',
            'Train Time (s)', 'Augmentations'
        ])

    def _load_config(self, config_path):
        if config_path is None:
            config_path = self.project_root / "src" / "config" / "config.yaml"
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        cfg = OmegaConf.load(config_path)
        cfg.data.root_dir = str(self.project_root / "data")
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

    def _get_class_distribution(self, targets):
        values, counts = np.unique(targets, return_counts=True)
        class_dist = {int(v): int(c) for v, c in zip(values, counts)}
        return class_dist.get(0, 0), class_dist.get(1, 0)

    def _plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        return fig

    def _train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        stats = {'loss': 0.0, 'correct': 0, 'total': 0}
        all_targets, all_preds, all_probs = [], [], []
        aug_count = 0

        for inputs, targets in train_loader:
            if hasattr(inputs, 'shape') and inputs.shape[0] > 1:
                aug_count += 1

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

        return loss, acc, precision, recall, f1, auc, all_targets, aug_count

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
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.training.learning_rate,
                                weight_decay=self.cfg.model.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.training.epochs)

        for epoch in range(self.cfg.training.epochs):
            start_time = time.time()
            train_loss, train_acc, train_prec, train_rec, train_f1, train_auc, train_targets, aug_count = self._train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_cm, val_targets = self._validate(val_loader, criterion)
            epoch_time = time.time() - start_time

            train_c0, train_c1 = self._get_class_distribution(train_targets)
            val_samples = len(val_targets)
            train_samples = len(train_targets)

            print(f"\nEpoch {epoch + 1}/{self.cfg.training.epochs} | Time: {epoch_time:.1f}s")
            print(f"Train: Loss={train_loss:.4f} Acc={train_acc:.2f}% Prec={train_prec:.2f} Rec={train_rec:.2f} F1={train_f1:.2f} AUC={train_auc:.2f}")
            print(f"Val:   Loss={val_loss:.4f} Acc={val_acc:.2f}% Prec={val_prec:.2f} Rec={val_rec:.2f} F1={val_f1:.2f} AUC={val_auc:.2f}")

            # Zapis do CSV
            self.metrics_writer.writerow([
                epoch + 1, train_samples, val_samples, train_c0, train_c1,
                train_loss, train_acc, train_prec, train_rec, train_f1, train_auc,
                val_loss, val_acc, val_prec, val_rec, val_f1, val_auc,
                round(epoch_time, 2), aug_count
            ])

            # TensorBoard
            self.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
            self.writer.add_scalars("Precision", {"train": train_prec, "val": val_prec}, epoch)
            self.writer.add_scalars("Recall", {"train": train_rec, "val": val_rec}, epoch)
            self.writer.add_scalars("F1", {"train": train_f1, "val": val_f1}, epoch)
            self.writer.add_scalars("AUC", {"train": train_auc, "val": val_auc}, epoch)
            self.writer.add_scalar("Augmentations", aug_count, epoch)
            self.writer.add_scalar("Epoch Time", epoch_time, epoch)
            self.writer.add_figure("Val Confusion Matrix", self._plot_confusion_matrix(val_cm), epoch)

            scheduler.step()

        self.metrics_file.close()
        self.writer.close()


if __name__ == "__main__":
    try:
        trainer = Trainer()
        trainer.train()
    except Exception as e:
        print(f"Error: {e}")
        raise
