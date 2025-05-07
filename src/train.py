import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from data_loader.dataset import HerringDataset
from models.model import HerringModel
from sklearn.metrics import precision_score, recall_score, f1_score


class Trainer:
    def __init__(self, config_path: str = None):
        """
        Inicjalizacja trenera dla struktury projektu HerringProject
        Args:
            config_path: Opcjonalna ścieżka do konfiguracji
        """
        # 1. Inicjalizacja ścieżek
        self.project_root = Path(__file__).parent.parent
        print(f"\nProject root: {self.project_root}")

        # 2. Ładowanie konfiguracji
        self.cfg = self._load_config(config_path)

        # 3. Inicjalizacja urządzenia
        self.device = self._init_device()
        print(f"Using device: {self.device}")

        # 4. Weryfikacja struktury danych
        self._validate_data_structure()

        # 5. Inicjalizacja komponentów
        self.model = HerringModel(self.cfg).to(self.device)
        self.data_loader = HerringDataset(self.cfg)
        self.writer = self._init_tensorboard()

    def _load_config(self, config_path):
        """Ładowanie konfiguracji z automatyczną korektą ścieżek"""
        if config_path is None:
            config_path = self.project_root / "src" / "config" / "config.yaml"

        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        cfg = OmegaConf.load(config_path)

        # Aktualizacja ścieżek w konfiguracji
        cfg.data.root_dir = str(self.project_root / "data")
        return cfg

    def _init_device(self):
        """Inicjalizacja urządzenia z pełną obsługą CPU/GPU"""
        if torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                _ = torch.tensor([1.0]).to(device)
                return device
            except Exception as e:
                print(f"Warning: GPU initialization failed. Falling back to CPU. Error: {str(e)}")

        return torch.device("cpu")

    def _validate_data_structure(self):
        """Pełna weryfikacja struktury katalogów danych"""
        print("\nValidating data structure...")

        train_dir = self.project_root / "data" / "train"
        val_dir = self.project_root / "data" / "val"

        required_dirs = {
            "train": train_dir,
            "val": val_dir
        }

        for name, dir_path in required_dirs.items():
            if not dir_path.exists():
                raise FileNotFoundError(
                    f"Directory '{name}' not found at: {dir_path}\n"
                    f"Expected structure:\n"
                    f"data/\n"
                    f"├── train/\n"
                    f"│   ├── 1/\n"
                    f"│   └── 2/\n"
                    f"└── val/\n"
                    f"    ├── 1/\n"
                    f"    └── 2/"
                )

            class_dirs = [d for d in dir_path.iterdir() if d.is_dir()]
            if not class_dirs:
                raise FileNotFoundError(
                    f"No class directories found in {dir_path}\n"
                    f"Each directory should contain subdirectories '1/' and '2/'"
                )

            print(f"Found {len(class_dirs)} class directories in {name}")

        print("Data structure validation passed successfully")

    def _init_tensorboard(self):
        """Inicjalizacja TensorBoard z obsługą błędów"""
        log_dir = self.project_root / "results" / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            return SummaryWriter(log_dir=str(log_dir))
        except Exception as e:
            print(f"Warning: TensorBoard initialization failed. Error: {str(e)}")
            return None

    def _train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Pojedyncza epoka treningowa"""
        self.model.train()
        stats = {'loss': 0.0, 'correct': 0, 'total': 0}

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            stats['loss'] += loss.item()
            _, predicted = outputs.max(1)
            stats['total'] += targets.size(0)
            stats['correct'] += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        epoch_loss = stats['loss'] / len(train_loader)
        epoch_acc = 100. * stats['correct'] / stats['total']

        # Oblicz inne metryki
        precision = precision_score(targets.cpu(), predicted.cpu(), average='binary')
        recall = recall_score(targets.cpu(), predicted.cpu(), average='binary')
        f1 = f1_score(targets.cpu(), predicted.cpu(), average='binary')

        if self.writer:
            self.writer.add_scalar("Loss/train", epoch_loss, epoch)
            self.writer.add_scalar("Accuracy/train", epoch_acc, epoch)
            self.writer.add_scalar("Precision/train", precision, epoch)
            self.writer.add_scalar("Recall/train", recall, epoch)
            self.writer.add_scalar("F1/train", f1, epoch)

        return epoch_loss, epoch_acc

    def _validate(self, val_loader, criterion, epoch):
        """Walidacja modelu"""
        self.model.eval()
        stats = {'loss': 0.0, 'correct': 0, 'total': 0}

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                stats['loss'] += loss.item()
                _, predicted = outputs.max(1)
                stats['total'] += targets.size(0)
                stats['correct'] += predicted.eq(targets).sum().item()

        val_loss = stats['loss'] / len(val_loader)
        val_acc = 100. * stats['correct'] / stats['total']

        # Oblicz inne metryki
        precision = precision_score(targets.cpu(), predicted.cpu(), average='binary')
        recall = recall_score(targets.cpu(), predicted.cpu(), average='binary')
        f1 = f1_score(targets.cpu(), predicted.cpu(), average='binary')

        if self.writer:
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar("Precision/val", precision, epoch)
            self.writer.add_scalar("Recall/val", recall, epoch)
            self.writer.add_scalar("F1/val", f1, epoch)

        return val_loss, val_acc

    def _save_checkpoint(self, epoch, optimizer, val_acc, class_names):
        """Zapis checkpointu modelu"""
        checkpoint_dir = self.project_root / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        model_name = self.cfg.model.base_model.replace('/', '_')
        checkpoint_path = checkpoint_dir / f"{model_name}_epoch_{epoch}_acc_{val_acc:.2f}.pth"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'class_names': class_names,
            'config': OmegaConf.to_container(self.cfg)
        }, str(checkpoint_path))

        return checkpoint_path

    def train(self):
        """Główna pętla treningowa"""

        train_loader, val_loader, class_names = self.data_loader.get_loaders()

        # Konfiguracja treningu
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.training.epochs
        )

        best_acc = 0.0
        for epoch in range(self.cfg.training.epochs):
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion, epoch)
            val_loss, val_acc = self._validate(val_loader, criterion, epoch)
            scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint_path = self._save_checkpoint(epoch + 1, optimizer, val_acc, class_names)
                print(f"Saved best model to: {checkpoint_path}")

        if self.writer:
            self.writer.close()


if __name__ == "__main__":
    try:
        trainer = Trainer()
        trainer.train()
    except Exception as e:
        print(f"Error during training initialization: {str(e)}")
        raise
