import os
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from collections import Counter, defaultdict
from omegaconf import DictConfig
from pathlib import Path
from utils.path_manager import PathManager
from utils.population_mapper import PopulationMapper    # DODANO
from PIL import Image

class HerringValDataset(Dataset):
    def __init__(self, image_folder, metadata, transform, population_mapper):
        self.image_folder = image_folder
        self.metadata = metadata
        self.transform = transform
        self.population_mapper = population_mapper
        self.valid_indices = [
            idx for idx, (path, _) in enumerate(self.image_folder.imgs)
            if self._is_valid(path)
        ]

    def _is_valid(self, path):
        fname = os.path.basename(path).strip().lower()
        meta = self.metadata.get(fname, (-9, -9))
        pop = meta[0]
        return pop in self.population_mapper.active_populations

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        path, _ = self.image_folder.imgs[real_idx]
        image = Image.open(path).convert("RGB")
        img_tensor = self.transform(image)
        filename = os.path.basename(path).strip().lower()
        pop, wiek = self.metadata.get(filename, (-9, -9))
        label = self.population_mapper.to_idx(pop)
        meta = {'populacja': pop, 'wiek': wiek}
        return img_tensor, label, meta

class HerringDataset:
    def __init__(self, config: DictConfig, population_mapper=None):
        self.cfg = config
        self.path_manager = PathManager(Path(__file__).parent.parent.parent, config)
        self.train_transform_base = self._get_base_transforms()
        self.train_transform_strong = self._get_strong_transforms()
        self.val_transform = self._get_val_transforms()
        self.metadata = self._load_metadata()
        self.class_counts = self._compute_class_counts()
        self.max_count = max(self.class_counts.values())
        self.augment_applied = defaultdict(int)
        self.active_populations = list(self.cfg.data.active_populations)
        self.population_mapper = population_mapper or PopulationMapper(self.active_populations)
        self._validate_labels()

    def _load_metadata(self):
        excel_path = self.path_manager.metadata_file()
        df = pd.read_excel(excel_path, engine="openpyxl")
        df["Wiek"] = pd.to_numeric(df["Wiek"], errors="coerce").fillna(-9).astype(int)
        df["Populacja"] = pd.to_numeric(df["Populacja"], errors="coerce").fillna(-9).astype(int)
        return {
            str(row["FileName"]).strip().lower(): (int(row["Populacja"]), int(row["Wiek"]))
            for _, row in df.iterrows()
        }

    def _compute_class_counts(self):
        counter = Counter()
        for key, (pop, wiek) in self.metadata.items():
            counter[(pop, wiek)] += 1
        return counter

    def _get_image_size(self):
        if self.cfg.get("multitask_model") and self.cfg.multitask_model.use:
            return self.cfg.multitask_model.backbone_model.image_size
        return self.cfg.base_model.image_size

    def _get_base_transforms(self):
        size = self._get_image_size()
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _get_strong_transforms(self):
        aug = self.cfg.augmentation
        size = self._get_image_size()
        return transforms.Compose([
            transforms.RandomRotation(aug.rotation),
            transforms.RandomResizedCrop(size, scale=tuple(aug.crop_scale)),
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
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.GaussianBlur(kernel_size=aug.gaussian_blur_kernel)
        ])

    def _get_val_transforms(self):
        size = self._get_image_size()
        return transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _validate_labels(self):
        data_root = self.path_manager.data_root()
        train_labels = sorted(os.listdir(data_root / 'train'))
        val_labels = sorted(os.listdir(data_root / 'val'))
        expected_labels = sorted([str(p) for p in self.active_populations])
        if train_labels != expected_labels or val_labels != expected_labels:
            raise ValueError(f"Niepoprawne etykiety: {train_labels}, {val_labels}")
        print(f"✔️ Etykiety klas poprawne {expected_labels}")

    def get_loaders(self):
        from torchvision import datasets
        data_root = self.path_manager.data_root()
        train_dir = data_root / 'train'
        val_dir = data_root / 'val'

        train_base = datasets.ImageFolder(str(train_dir))
        train_base.transform = self.train_transform_base
        val_base = datasets.ImageFolder(str(val_dir))

        val_set = HerringValDataset(val_base, self.metadata, self.val_transform, self.population_mapper)
        train_set = HerringValDataset(train_base, self.metadata, self.train_transform_base, self.population_mapper)

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

        # DEBUG: sprawdź czy batch daje populacje takie jak z excela
        for batch in train_loader:
            imgs, labels, metas = batch
            print("DEBUG batch klasy indeksy (labels):", labels)
            if isinstance(metas, dict) and 'populacja' in metas:
                print("DEBUG batch meta populacje (z excela):", metas['populacja'])
                print("DEBUG batch meta wiek:", metas['wiek'])
            else:
                print("DEBUG batch meta populacje (z excela):", metas)
            print("DEBUG batch klasy biologiczne:", [self.population_mapper.to_pop(idx) for idx in labels])
            break

        return train_loader, val_loader, self.active_populations

