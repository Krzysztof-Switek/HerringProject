import os
import torch

class AugmentWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, metadata, class_counts, max_count, transform_base, transform_strong, augment_applied, active_populations):
        print("DEBUG [augment_utils.py::__init__] Tworzenie AugmentWrapper, liczba próbek w base_dataset:", len(base_dataset.samples))
        self.base_dataset = base_dataset
        self.metadata = metadata
        self.class_counts = class_counts
        self.max_count = max_count
        self.transform_base = transform_base
        self.transform_strong = transform_strong
        self.augment_applied = augment_applied
        self.active_populations = active_populations
        self._printed_augmentation_notice = False

        # FILTRUJ INDEKSY tylko dla populacji z configa
        self.valid_indices = [
            idx for idx, (path, label) in enumerate(self.base_dataset.samples)
            if self._is_valid(path)
        ]
        print("DEBUG [augment_utils.py::__init__] valid_indices:", len(self.valid_indices))
        if len(self.valid_indices) < 10:
            print("DEBUG [augment_utils.py::__init__] Pierwsze valid_indices:", self.valid_indices[:10])
        for idx in self.valid_indices[:5]:
            path, label = self.base_dataset.samples[idx]
            print(f"DEBUG [augment_utils.py::__init__] valid path: {path}, label: {label}")

    def _is_valid(self, path):
        fname = os.path.basename(path).strip().lower()
        meta = self.metadata.get(fname, (-9, -9))   # -9 oznacza brak/nieznany
        pop = meta[0]
        print(f"DEBUG [augment_utils.py::_is_valid] fname: {fname}, meta: {meta}, pop: {pop}, active_populations: {self.active_populations}")
        return pop in self.active_populations

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        path, label = self.base_dataset.samples[real_idx]
        image = self.base_dataset.loader(path)

        fname = os.path.basename(path).strip().lower()

        if fname not in self.metadata:
            print(f"DEBUG [augment_utils.py::__getitem__] ⚠️ Nie znaleziono metadanych dla pliku: {fname}")
            transform = self.transform_base
            return transform(image), label, {"populacja": torch.tensor(-9), "wiek": torch.tensor(-9)}

        pop, wiek = self.metadata[fname]

        if pop not in self.active_populations:
            print(f"DEBUG [augment_utils.py::__getitem__] BŁĄD: {fname}, pop: {pop}, active_populations: {self.active_populations}")
            raise ValueError(f"Plik {fname} ma niedozwoloną populację: {pop} (dozwolone: {self.active_populations})")

        count = self.class_counts.get((pop, wiek), 0)
        desired_total = self.max_count
        augment_needed = max(0, desired_total - count)
        prob = min(1.0, augment_needed / desired_total)

        if torch.rand(1).item() < prob:
            self.augment_applied[(pop, wiek)] += 1
            transform = self.transform_strong

            if not self._printed_augmentation_notice:
                print("DEBUG [augment_utils.py::__getitem__] ✨ Augmentacja w toku dla klas o małej liczności...")
                self._printed_augmentation_notice = True
        else:
            transform = self.transform_base

        return transform(image), label, {"populacja": torch.tensor(pop), "wiek": torch.tensor(wiek)}
