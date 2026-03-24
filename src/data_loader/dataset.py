from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import get_eval_transform, get_image_size, get_train_transform
from ..engine.augment_utils import AugmentWrapper
from ..utils.path_manager import PathManager
from ..utils.population_mapper import PopulationMapper
from ..utils.config_helpers import get_augmentation_mode


class HerringValDataset(Dataset):
    def __init__(self, image_folder, metadata, transform, population_mapper: PopulationMapper):
        self.image_folder = image_folder
        self.metadata = metadata
        self.transform = transform
        self.population_mapper = population_mapper
        self.valid_indices = [
            idx for idx, (path, _) in enumerate(self.image_folder.imgs)
            if self._is_valid(path)
        ]

    def _is_valid(self, path: str) -> bool:
        fname = os.path.basename(path).strip().lower()
        meta = self.metadata.get(fname, (-9, -9))
        pop = meta[0]
        return pop in self.population_mapper.active_populations

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        real_idx = self.valid_indices[idx]
        path, _ = self.image_folder.imgs[real_idx]
        image = Image.open(path).convert("RGB")
        img_tensor = self.transform(image)

        filename = os.path.basename(path).strip().lower()
        pop, wiek = self.metadata.get(filename, (-9, -9))
        label = self.population_mapper.to_idx(pop)

        meta = {
            "populacja": pop,
            "wiek": wiek,
        }
        return img_tensor, label, meta


class HerringDataset:
    def __init__(
        self,
        config: DictConfig,
        population_mapper: PopulationMapper | None = None,
        metadata_override: dict | None = None,
        skip_validation: bool = False,
    ):
        self.cfg = config
        self.path_manager = PathManager(Path(__file__).parent.parent.parent, config)

        self.active_populations = list(self.cfg.data.active_populations)
        self.population_mapper = population_mapper or PopulationMapper(self.active_populations)

        self.train_transform_base = get_train_transform(self.cfg, mode="base")
        self.train_transform_strong = get_train_transform(self.cfg, mode="strong")
        self.val_transform = get_eval_transform(self.cfg)

        if metadata_override is not None:
            self.metadata = metadata_override
        else:
            self.metadata = self._load_metadata()

        self.class_counts = self._compute_class_counts()
        self.max_count = max(self.class_counts.values()) if self.class_counts else 0
        self.augment_applied = defaultdict(int)

        if not skip_validation:
            self._validate_labels()

    def _load_metadata(self):
        excel_path = self.path_manager.metadata_file()
        df = pd.read_excel(excel_path, engine="openpyxl")

        required_columns = {"FileName", "Populacja", "Wiek"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Brakuje kolumn w metadata: {sorted(missing)}")

        df["Wiek"] = pd.to_numeric(df["Wiek"], errors="coerce").fillna(-9).astype(int)
        df["Populacja"] = pd.to_numeric(df["Populacja"], errors="coerce").fillna(-9).astype(int)

        return {
            str(row["FileName"]).strip().lower(): (int(row["Populacja"]), int(row["Wiek"]))
            for _, row in df.iterrows()
        }

    def _compute_class_counts(self):
        counter = Counter()
        for _file_name, (pop, wiek) in self.metadata.items():
            if pop in self.active_populations:
                counter[(pop, wiek)] += 1
        return counter

    def _get_image_size(self) -> int:
        return get_image_size(self.cfg)

    def _get_train_transform(self):
        mode = get_augmentation_mode(self.cfg)
        if mode == "strong":
            return self.train_transform_strong
        return self.train_transform_base

    def _validate_labels(self):
        data_root = self.path_manager.data_root()
        train_dir = data_root / "train"
        val_dir = data_root / "val"

        if not train_dir.exists():
            raise FileNotFoundError(f"Nie znaleziono katalogu treningowego: {train_dir}")
        if not val_dir.exists():
            raise FileNotFoundError(f"Nie znaleziono katalogu walidacyjnego: {val_dir}")

        train_labels = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        val_labels = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
        expected_labels = sorted([str(p) for p in self.active_populations])

        if train_labels != expected_labels:
            raise ValueError(
                f"Niepoprawne etykiety w train/: {train_labels}; oczekiwano: {expected_labels}"
            )
        if val_labels != expected_labels:
            raise ValueError(
                f"Niepoprawne etykiety w val/: {val_labels}; oczekiwano: {expected_labels}"
            )

    def get_loaders(self):
        from torchvision import datasets

        data_root = self.path_manager.data_root()
        train_dir = data_root / "train"
        val_dir = data_root / "val"

        train_base = datasets.ImageFolder(str(train_dir))
        val_base = datasets.ImageFolder(str(val_dir))

        train_set = AugmentWrapper(
            base_dataset=train_base,
            metadata=self.metadata,
            class_counts=self.class_counts,
            max_count=self.max_count,
            transform_base=self.train_transform_base,
            transform_strong=self.train_transform_strong,
            augment_applied=self.augment_applied,
            population_mapper=self.population_mapper,
        )
        val_set = HerringValDataset(
            val_base,
            self.metadata,
            self.val_transform,
            self.population_mapper,
        )

        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            val_set,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        return train_loader, val_loader, self.active_populations
