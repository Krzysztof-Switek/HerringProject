from torch.utils.data import Dataset
import os
from PIL import Image

class HerringValDataset(Dataset):
    def __init__(self, image_folder, metadata, transform, active_populations):
        print("DEBUG [dataset.py::__init__] Tworzenie HerringValDataset, liczba obrazów:", len(image_folder.imgs))
        self.image_folder = image_folder
        self.metadata = metadata
        self.transform = transform
        self.active_populations = list(active_populations)

        self.valid_indices = [
            idx for idx, (path, label) in enumerate(self.image_folder.imgs)
            if self._is_valid(path)
        ]
        print("DEBUG [dataset.py::__init__] valid_indices:", len(self.valid_indices))
        if len(self.valid_indices) < 10:
            print("DEBUG [dataset.py::__init__] Pierwsze valid_indices:", self.valid_indices[:10])
        for idx in self.valid_indices[:5]:
            path, label = self.image_folder.imgs[idx]
            print(f"DEBUG [dataset.py::__init__] valid path: {path}, label: {label}")

    def _is_valid(self, path):
        fname = os.path.basename(path).strip().lower()
        meta = self.metadata.get(fname, (-9, -9))
        pop = meta[0]
        print(f"DEBUG [dataset.py::_is_valid] fname: {fname}, meta: {meta}, pop: {pop}, active_populations: {self.active_populations}")
        return pop in self.active_populations

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        path, label = self.image_folder.imgs[real_idx]
        image = Image.open(path).convert("RGB")
        img_tensor = self.transform(image)
        filename = os.path.basename(path).strip().lower()
        pop, wiek = self.metadata.get(filename, (-9, -9))
        meta = {'populacja': pop, 'wiek': wiek}
        return img_tensor, label, meta
