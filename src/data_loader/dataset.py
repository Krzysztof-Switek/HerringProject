import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from omegaconf import DictConfig


class HerringDataset:
    def __init__(self, config: DictConfig):
        """
        Inicjalizacja ładowania danych z walidacją ścieżek

        Args:
            config: Konfiguracja projektu (omegaconf.DictConfig)
        """
        self.cfg = config.data
        self.transform = self._get_transforms()
        self._validate_paths()

    def _get_transforms(self) -> transforms.Compose:
        """
        Definiuje transformacje dla danych treningowych i walidacyjnych

        Returns:
            transforms.Compose: Zestaw transformacji
        """
        return transforms.Compose([
            transforms.Resize(self.cfg.image_size),
            transforms.CenterCrop(self.cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def _validate_paths(self):
        """Weryfikuje istnienie wymaganych katalogów i klas"""
        required_dirs = [
            os.path.join(self.cfg.root_dir, self.cfg.train),
            os.path.join(self.cfg.root_dir, self.cfg.val)
        ]

        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(
                    f"Katalog danych {dir_path} nie istnieje. "
                    f"Upewnij się że wykonałeś podział danych."
                )

            # Sprawdź czy są podkatalogi klas
            subdirs = [d for d in os.listdir(dir_path)
                       if os.path.isdir(os.path.join(dir_path, d))]
            if not subdirs:
                raise RuntimeError(
                    f"Brak podkatalogów klas w {dir_path}. "
                    f"Oczekiwano strukturę: {dir_path}/1/, {dir_path}/2/"
                )

    def get_loaders(self) -> tuple:
        """
        Przygotowuje DataLoadery dla danych treningowych i walidacyjnych

        Returns:
            tuple: (train_loader, val_loader, class_names)
        """
        train_dir = os.path.join(self.cfg.root_dir, self.cfg.train)
        val_dir = os.path.join(self.cfg.root_dir, self.cfg.val)

        train_set = datasets.ImageFolder(train_dir, self.transform)
        val_set = datasets.ImageFolder(val_dir, self.transform)

        # Weryfikacja spójności klas
        if train_set.classes != val_set.classes:
            raise ValueError(
                "Niespójne klasy między zestawem treningowym a walidacyjnym. "
                f"Train classes: {train_set.classes}, Val classes: {val_set.classes}"
            )

        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_loader = DataLoader(
            val_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )

        return train_loader, val_loader, train_set.classes

    def get_class_distribution(self) -> dict:
        """
        Zwraca rozkład klas w danych treningowych i walidacyjnych

        Returns:
            dict: {
                'train': {class1: count, class2: count},
                'val': {class1: count, class2: count}
            }
        """
        train_dir = os.path.join(self.cfg.root_dir, self.cfg.train)
        val_dir = os.path.join(self.cfg.root_dir, self.cfg.val)

        train_dist = {
            cls: len(os.listdir(os.path.join(train_dir, cls)))
            for cls in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, cls))
        }

        val_dist = {
            cls: len(os.listdir(os.path.join(val_dir, cls)))
            for cls in os.listdir(val_dir)
            if os.path.isdir(os.path.join(val_dir, cls))
        }

        return {'train': train_dist, 'val': val_dist}