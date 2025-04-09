import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from omegaconf import DictConfig


class HerringDataset:
    def __init__(self, config: DictConfig):
        self.cfg = config.data
        self.transform = self._get_transforms()

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def get_loaders(self):
        train_dir = f"{self.cfg.root_dir}/{self.cfg.train}"
        val_dir = f"{self.cfg.root_dir}/{self.cfg.val}"

        train_set = datasets.ImageFolder(train_dir, self.transform)
        val_set = datasets.ImageFolder(val_dir, self.transform)

        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=2
        )

        val_loader = DataLoader(
            val_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=2
        )

        return train_loader, val_loader, train_set.classes