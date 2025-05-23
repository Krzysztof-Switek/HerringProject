import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from omegaconf import DictConfig


class HerringDataset:
    def __init__(self, config: DictConfig):
        """
        Inicjalizacja ładowania danych dla struktury projektu HerringProject

        Args:
            config: Konfiguracja z pliku config.yaml
        """
        self.cfg = config.data
        self._validate_paths()
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()

        # Sprawdzamy, czy etykiety w zbiorze danych zgadzają się z nazwami katalogów
        self._validate_labels()

    def _get_train_transforms(self) -> transforms.Compose:
        """Transformacje z augmentacją dla danych treningowych"""
        return transforms.Compose([
            transforms.RandomRotation(30),  # Losowy obrót do 30 stopni
            transforms.RandomResizedCrop(self.cfg.image_size, scale=(0.8, 1.0)),  # Losowe przycięcie i skalowanie
            transforms.RandomHorizontalFlip(p=0.5),  # Losowe odbicie w poziomie
            transforms.RandomVerticalFlip(p=0.5),  # Losowe odbicie w pionie
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10
            ),
            transforms.Resize((self.cfg.image_size, self.cfg.image_size)),  # Zmiana rozmiaru obrazu
            transforms.ToTensor(),  # Konwersja na tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.GaussianBlur(kernel_size=3)  # Dodanie rozmycia Gaussa
        ])

    def _get_val_transforms(self) -> transforms.Compose:
        """Transformacje bez augmentacji dla danych walidacyjnych"""
        return transforms.Compose([
            transforms.Resize(self.cfg.image_size),
            transforms.CenterCrop(self.cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def _validate_paths(self):
        """Weryfikacja struktury katalogów danych"""
        base_path = os.path.join(os.path.dirname(__file__), '../../..', self.cfg.root_dir)
        required_paths = {
            'train': os.path.join(base_path, 'train'),
            'val': os.path.join(base_path, 'val')
        }

        for name, path in required_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Nie znaleziono katalogu {name}: {path}\n"
                    f"Pełna ścieżka: {os.path.abspath(path)}"
                )

            if not os.listdir(path):
                raise RuntimeError(f"Katalog {name} jest pusty: {path}")

    def _validate_labels(self):
        """Weryfikacja zgodności etykiet z nazwami katalogów"""
        base_path = os.path.join(os.path.dirname(__file__), '../../..', self.cfg.root_dir)
        train_dir = os.path.join(base_path, 'train')
        val_dir = os.path.join(base_path, 'val')

        # Etykiety to nazwy katalogów w 'train' i 'val'
        train_labels = sorted(os.listdir(train_dir))
        val_labels = sorted(os.listdir(val_dir))

        # Sprawdzamy, czy etykiety są zgodne z nazwami katalogów
        expected_labels = ['0', '1']

        if train_labels != expected_labels:
            raise ValueError(
                f"Niepoprawne etykiety w katalogu 'train'. Oczekiwano {expected_labels}, ale znaleziono {train_labels}")

        if val_labels != expected_labels:
            raise ValueError(
                f"Niepoprawne etykiety w katalogu 'val'. Oczekiwano {expected_labels}, ale znaleziono {val_labels}")

        print(f"Zgodność etykiet w katalogach: 'train' i 'val' jest poprawna.")

    def get_loaders(self) -> tuple:
        """
        Przygotowuje DataLoadery dla danych treningowych i walidacyjnych

        Returns:
            tuple: (train_loader, val_loader, class_names)
        """
        base_path = os.path.join(os.path.dirname(__file__), '../../..')
        train_dir = os.path.join(base_path, self.cfg.root_dir, 'train')
        val_dir = os.path.join(base_path, self.cfg.root_dir, 'val')

        train_set = datasets.ImageFolder(train_dir, self.train_transform)
        val_set = datasets.ImageFolder(val_dir, self.val_transform)

        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

        return train_loader, val_loader, train_set.classes

    def show_stats(self):
        """Wyświetla statystyki danych"""
        base_path = os.path.join(os.path.dirname(__file__), '../../..')
        train_dir = os.path.join(base_path, self.cfg.root_dir, 'train')
        val_dir = os.path.join(base_path, self.cfg.root_dir, 'val')

        train_counts = {
            cls: len(os.listdir(os.path.join(train_dir, cls)))
            for cls in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, cls))
        }

        val_counts = {
            cls: len(os.listdir(os.path.join(val_dir, cls)))
            for cls in os.listdir(val_dir)
            if os.path.isdir(os.path.join(val_dir, cls))
        }

        print("\nStatystyki danych:")
        print(f"Treningowe: {sum(train_counts.values())} obrazów")
        for cls, count in train_counts.items():
            print(f"  Klasa {cls}: {count}")

        print(f"\nWalidacyjne: {sum(val_counts.values())} obrazów")
        for cls, count in val_counts.items():
            print(f"  Klasa {cls}: {count}")
