import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from omegaconf import DictConfig
from utils.path_manager import PathManager

class HerringCustomDataset(Dataset):
    def __init__(self, root_dir, metadata, transform, valid_pops=[1, 2]):  # <--- możesz podać dowolne populacje
        self.root_dir = root_dir
        self.metadata = metadata
        self.transform = transform
        self.valid_pops = valid_pops  # 🟢 pozwala ograniczyć tylko do 1,2, jeśli chcesz
        self.imgs = []

        # --- 🟢 Najważniejsze: Budujemy listę (ścieżka, populacja) tylko dla właściwych populacji ---
        for pop in self.valid_pops:
            pop_dir = os.path.join(root_dir, str(pop))
            if os.path.isdir(pop_dir):
                for fname in os.listdir(pop_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.imgs.append((os.path.join(pop_dir, fname), pop))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, pop = self.imgs[idx]   # 🟢 label = rzeczywista populacja (1/2/3/0)
        image = Image.open(path).convert("RGB")
        img_tensor = self.transform(image)
        filename = os.path.basename(path).strip().lower()
        wiek = self.metadata.get(filename, (-1, -9))[1]
        meta = {"populacja": pop, "wiek": wiek}
        return img_tensor, pop, meta   # 🟢 zawsze pop, nie (0/1), tylko (1/2/...)

# -------------------------------------------------------------

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
        df["Wiek"] = df["Wiek"].fillna(-9).astype(int)
        df["Populacja"] = df["Populacja"].astype(int)
        # --- 🟢 Tu NIE filtrujemy populacji – może być 0,1,2,3 ---
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
        # --- 🟢 Sprawdza obecność folderów 1,2,3,0 ---
        folder_labels = sorted([f for f in os.listdir(data_root / 'train') if os.path.isdir(data_root / 'train' / f)])
        print(f"Etykiety folderów train: {folder_labels}")
        # Jeśli oczekujesz tylko [1,2], zmień poniżej
        # if folder_labels != ['1', '2']: ... raise
        # Jeśli chcesz obsłużyć 0/1/2/3, zmień na poniższe:
        # expected_labels = ['0', '1', '2', '3']
        # if any(lbl not in folder_labels for lbl in expected_labels):
        #     raise ValueError(...)
        print("✔️ Etykiety folderów są zgodne z populacjami biologicznymi")

    def get_loaders(self):
        data_root = self.path_manager.data_root()
        train_dir = data_root / 'train'
        val_dir = data_root / 'val'

        # --- 🟢 ZAMIANA: własny dataset, nie ImageFolder ---
        train_set = HerringCustomDataset(
            str(train_dir), self.metadata, self.train_transform_base, valid_pops=[1, 2]
        )
        val_set = HerringCustomDataset(
            str(val_dir), self.metadata, self.val_transform, valid_pops=[1, 2]
        )

        from engine.augment_utils import AugmentWrapper
        train_set = AugmentWrapper(
            base_dataset=train_set,
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

        return train_loader, val_loader, [1, 2]  # 🟢 lista etykiet (tu: [1,2], możesz dodać 0/3 jeśli chcesz)
