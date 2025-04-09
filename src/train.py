import os
import torch
from omegaconf import OmegaConf
from data_loader.dataset import HerringDataset
from models.model import HerringModel


class Trainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg = OmegaConf.load(config_path)
        self.device = torch.device(self.cfg.training.device)

        # Sprawdzenie istnienia katalogów
        self._validate_paths()

        self.model = HerringModel(self.cfg).to(self.device)
        self.data_loader = HerringDataset(self.cfg)

    def _validate_paths(self):
        """Weryfikacja poprawności ścieżek danych"""
        required_dirs = [
            os.path.join(self.cfg.data.root_dir, self.cfg.data.train),
            os.path.join(self.cfg.data.root_dir, self.cfg.data.val)
        ]

        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(
                    f"Katalog {dir_path} nie istnieje. "
                    f"Upewnij się że wykonałeś split danych."
                )
            if not os.listdir(dir_path):
                raise RuntimeError(
                    f"Katalog {dir_path} jest pusty. "
                    f"Sprawdź podział danych."
                )

    def train(self):
        train_loader, val_loader, class_names = self.data_loader.get_loaders()

        # Przykładowa pętla treningowa
        for epoch in range(self.cfg.training.epochs):
            self.model.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                # ... trening ...

            # Walidacja
            self.model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    # ... walidacja ...


if __name__ == "__main__":
    trainer = Trainer()
    print("Rozpoczynanie treningu...")
    trainer.train()