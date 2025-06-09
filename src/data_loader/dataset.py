import os
import torch
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from collections import Counter, defaultdict
from omegaconf import DictConfig
from pathlib import Path

class AugmentWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, metadata, class_counts, max_count, transform_base, transform_strong, augment_applied):
        self.base_dataset = base_dataset
        self.metadata = metadata
        self.class_counts = class_counts
        self.max_count = max_count
        self.transform_base = transform_base
        self.transform_strong = transform_strong
        self.augment_applied = augment_applied

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        path, label = self.base_dataset.samples[index]
        image = self.base_dataset.loader(path)

        # 🔧 DODANE: lepsza normalizacja nazwy pliku
        fname = os.path.basename(path).strip().replace(" ", "_").lower()

        # 🔍 DODANE: test obecności w metadanych
        if fname not in self.metadata:
            print(f"⚠️ Nie znaleziono metadanych dla pliku: {fname}")
            transform = self.transform_base
            return transform(image), label, {"populacja": torch.tensor(-1), "wiek": torch.tensor(-1)}

        pop, wiek = self.metadata[fname]
        count = self.class_counts.get((pop, wiek), 0)

        # 🔧 DODANE: bezpieczne minimum count
        desired_total = self.max_count
        augment_needed = max(0, desired_total - count)
        prob = min(1.0, augment_needed / desired_total)

        #prob = 0.5                                                  # 🔧 TEST: stałe prawdopodobieństwo augmentacji

        if torch.rand(1).item() < prob:
            self.augment_applied[(pop, wiek)] += 1
            transform = self.transform_strong
            # 🔍 logowanie augmentacji
            if self.augment_applied[(pop, wiek)] < 5:  # ogranicz do kilku logów
                print(f"✨ Augmentacja dla ({pop}, {wiek}) - count: {count}, prob: {prob:.2f}")
        else:
            transform = self.transform_base

        return transform(image), label, {"populacja": torch.tensor(pop), "wiek": torch.tensor(wiek)}



class HerringDataset:
    def __init__(self, config: DictConfig):
        self.cfg = config.data
        self._validate_paths()
        self.train_transform_base = self._get_base_transforms()
        self.train_transform_strong = self._get_strong_transforms()
        self.val_transform = self._get_val_transforms()
        self.metadata = self._load_metadata()
        self.class_counts = self._compute_class_counts()
        self.max_count = max(self.class_counts.values())
        self.augment_applied = defaultdict(int)

        print(f"\n📊 Największa liczność klas (populacja, wiek): {self.max_count}")
        self._validate_labels()

    def _load_metadata(self):
        excel_path = Path(self.cfg.metadata_file)
        if not excel_path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku Excel: {excel_path}")

        df = pd.read_excel(excel_path, engine="openpyxl")
        if not all(col in df.columns for col in ["FileName", "Populacja", "Wiek"]):
            raise ValueError("Plik Excel musi zawierać kolumny: 'FileName', 'Populacja', 'Wiek'.")

        df = df[df["Populacja"].isin([1, 2])].copy()
        df["Wiek"] = df["Wiek"].fillna(-9).astype(int)
        df["Populacja"] = df["Populacja"].astype(int)

        return {
            str(row["FileName"]).strip().lower(): (int(row["Populacja"]), int(row["Wiek"]))
            for _, row in df.iterrows()
        }

    def _compute_class_counts(self):
        counter = Counter()
        for key, (pop, wiek) in self.metadata.items():
            counter[(pop, wiek)] += 1
        return counter

    def _get_base_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _get_strong_transforms(self):
        return transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(self.cfg.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.GaussianBlur(kernel_size=3)
        ])

    def _get_val_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.cfg.image_size),
            transforms.CenterCrop(self.cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _validate_paths(self):
        base_path = os.path.join(os.path.dirname(__file__), '../../..', self.cfg.root_dir)
        required_paths = {
            'train': os.path.join(base_path, 'train'),
            'val': os.path.join(base_path, 'val')
        }
        for name, path in required_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Brak katalogu {name}: {path}")
            if not os.listdir(path):
                raise RuntimeError(f"Katalog {name} jest pusty: {path}")

    def _validate_labels(self):
        base_path = os.path.join(os.path.dirname(__file__), '../../..', self.cfg.root_dir)
        train_labels = sorted(os.listdir(os.path.join(base_path, 'train')))
        val_labels = sorted(os.listdir(os.path.join(base_path, 'val')))
        expected_labels = ['1', '2']
        if train_labels != expected_labels or val_labels != expected_labels:
            raise ValueError(f"Niepoprawne etykiety: {train_labels}, {val_labels}")
        print("✔️ Etykiety klas poprawne (1 i 2)")

    def get_loaders(self):
        base_path = os.path.join(os.path.dirname(__file__), '../../..')
        train_dir = os.path.join(base_path, self.cfg.root_dir, 'train')
        val_dir = os.path.join(base_path, self.cfg.root_dir, 'val')

        train_base = datasets.ImageFolder(train_dir)
        val_set = datasets.ImageFolder(val_dir, transform=self.val_transform)

        train_set = AugmentWrapper(
            base_dataset=train_base,
            metadata=self.metadata,
            class_counts=self.class_counts,
            max_count=self.max_count,
            transform_base=self.train_transform_base,
            transform_strong=self.train_transform_strong,
            augment_applied=self.augment_applied
        )

        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        return train_loader, val_loader, train_base.classes

    ###
