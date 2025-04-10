import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from data_loader.dataset import HerringDataset
from models.model import HerringModel


class Trainer:
    def __init__(self, config_path: str = "../src/config/config.yaml"):
        """
        Inicjalizacja trenera dla struktury HerringProject

        Args:
            config_path: Ścieżka do pliku konfiguracyjnego (względem lokalizacji train.py)
        """
        # Ładowanie konfiguracji
        self.cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), config_path))
        self.device = torch.device(self.cfg.training.device)

        # Inicjalizacja komponentów
        self._init_directories()
        self.model = HerringModel(self.cfg).to(self.device)
        self.data_loader = HerringDataset(self.cfg)
        self.writer = SummaryWriter(log_dir="../results/logs")

        # Ścieżki bezwzględne dla przejrzystości
        self.train_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "..",
            self.cfg.data.root_dir,
            self.cfg.data.train
        ))
        self.val_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "..",
            self.cfg.data.root_dir,
            self.cfg.data.val
        ))

    def _init_directories(self):
        """Inicjalizacja wymaganych katalogów"""
        os.makedirs(os.path.join(os.path.dirname(__file__), "../checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(__file__), "../results/logs"), exist_ok=True)

    def _validate_paths(self):
        """Weryfikacja struktury danych"""
        if not os.path.exists(self.train_dir):
            raise FileNotFoundError(f"Nie znaleziono katalogu treningowego: {self.train_dir}")
        if not os.path.exists(self.val_dir):
            raise FileNotFoundError(f"Nie znaleziono katalogu walidacyjnego: {self.val_dir}")

    def _train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Pojedyncza epoka treningowa"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        # Logowanie do TensorBoard
        self.writer.add_scalar("Loss/train", epoch_loss, epoch)
        self.writer.add_scalar("Accuracy/train", epoch_acc, epoch)

        return epoch_loss, epoch_acc

    def _validate(self, val_loader, criterion, epoch):
        """Walidacja modelu"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * correct / total

        # Logowanie do TensorBoard
        self.writer.add_scalar("Loss/val", val_loss, epoch)
        self.writer.add_scalar("Accuracy/val", val_acc, epoch)

        return val_loss, val_acc

    def train(self):
        """Główna pętla treningowa"""
        self._validate_paths()
        train_loader, val_loader, class_names = self.data_loader.get_loaders()

        # Konfiguracja treningu
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.training.epochs)

        print(f"Rozpoczynanie treningu na {len(train_loader.dataset)} obrazach treningowych")
        print(f"Walidacja na {len(val_loader.dataset)} obrazach")
        print(f"Klasy: {class_names}")

        best_acc = 0.0
        for epoch in range(self.cfg.training.epochs):
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion, epoch)
            val_loss, val_acc = self._validate(val_loader, criterion, epoch)
            scheduler.step()

            print(f"Epoch {epoch + 1}/{self.cfg.training.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Zapis checkpointu
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint_path = os.path.join(
                    os.path.dirname(__file__),
                    "../checkpoints",
                    f"best_model_epoch_{epoch + 1}_acc_{val_acc:.2f}.pth"
                )
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_names': class_names
                }, checkpoint_path)
                print(f"Zapisano najlepszy model: {checkpoint_path}")

        self.writer.close()


if __name__ == "__main__":
    print("Inicjalizacja trenera...")
    trainer = Trainer()

    print("\nStatystyki danych:")
    trainer.data_loader.show_stats()

    print("\nRozpoczynanie treningu...")
    trainer.train()