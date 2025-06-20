import os
import torch
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from collections import Counter, defaultdict
from omegaconf import DictConfig
from pathlib import Path
from utils.path_manager import PathManager
from engine.augment_utils import AugmentWrapper


class HerringDataset:
    def __init__(self, config: DictConfig):
        self.cfg = config
        self.path_manager = PathManager(Path(__file__).parent.parent.parent, config)
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
        excel_path = self.path_manager.metadata_file()
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
            transforms.Resize((self.cfg.base_model.image_size, self.cfg.base_model.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _get_strong_transforms(self):  # 🔧 ZMODYFIKOWANE: korzysta z self.cfg.augmentation
        aug = self.cfg.augmentation
        return transforms.Compose([
            transforms.RandomRotation(aug.rotation),
            transforms.RandomResizedCrop(self.cfg.base_model.image_size, scale=tuple(aug.crop_scale)),
            transforms.RandomHorizontalFlip(p=aug.hflip_prob),
            transforms.RandomVerticalFlip(p=aug.vflip_prob),
            transforms.ColorJitter(
                brightness=aug.brightness,
                contrast=aug.contrast,
                saturation=aug.saturation,
                hue=aug.hue
            ),
            transforms.RandomAffine(
                degrees=aug.affine_degrees,
                translate=tuple(aug.affine_translate),
                scale=tuple(aug.affine_scale),
                shear=aug.affine_shear
            ),
            transforms.Resize((self.cfg.base_model.image_size, self.cfg.base_model.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.GaussianBlur(kernel_size=aug.gaussian_blur_kernel)
        ])

    def _get_val_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.cfg.base_model.image_size),
            transforms.CenterCrop(self.cfg.base_model.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _validate_labels(self):
        data_root = self.path_manager.data_root()
        train_labels = sorted(os.listdir(data_root / 'train'))
        val_labels = sorted(os.listdir(data_root / 'val'))
        expected_labels = ['1', '2']
        if train_labels != expected_labels or val_labels != expected_labels:
            raise ValueError(f"Niepoprawne etykiety: {train_labels}, {val_labels}")
        print("✔️ Etykiety klas poprawne (1 i 2)")

    def get_loaders(self):
        data_root = self.path_manager.data_root()
        train_dir = data_root / 'train'
        val_dir = data_root / 'val'

        train_base = datasets.ImageFolder(str(train_dir))
        val_set = datasets.ImageFolder(str(val_dir), transform=self.val_transform)

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
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_set,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        return train_loader, val_loader, train_base.classes
