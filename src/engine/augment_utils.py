import os
import torch
from utils.population_mapper import PopulationMapper   # [DODANO] importujemy mappera

class AugmentWrapper(torch.utils.data.Dataset):
    # ⬇️ [ZMIANA] Dodajemy population_mapper zamiast active_populations
    def __init__(self, base_dataset, metadata, class_counts, max_count, transform_base, transform_strong, augment_applied, population_mapper):
        self.base_dataset = base_dataset
        self.metadata = metadata
        self.class_counts = class_counts
        self.max_count = max_count
        self.transform_base = transform_base
        self.transform_strong = transform_strong
        self.augment_applied = augment_applied
        self.population_mapper = population_mapper      # [DODANO] zamiast active_populations
        self._printed_augmentation_notice = False

        # [USUNIĘTO] self.active_populations = active_populations

        self.valid_indices = [
            idx for idx, (path, _) in enumerate(self.base_dataset.samples)
            if self._is_valid(path)
        ]

    def _is_valid(self, path):
        fname = os.path.basename(path).strip().lower()
        meta = self.metadata.get(fname, (-9, -9))
        pop = meta[0]
        return pop in self.population_mapper.active_populations   # [ZAMIANA] filtracja przez population_mapper

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        path, _ = self.base_dataset.samples[real_idx]      # [ZAMIANA] _ zamiast label
        image = self.base_dataset.loader(path)

        fname = os.path.basename(path).strip().lower()

        if fname not in self.metadata:
            print(f"⚠️ Nie znaleziono metadanych dla pliku: {fname}")
            transform = self.transform_base
            return transform(image), -1, {"populacja": torch.tensor(-9), "wiek": torch.tensor(-9)}

        pop, wiek = self.metadata[fname]

        if pop not in self.population_mapper.active_populations:
            raise ValueError(f"Plik {fname} ma niedozwoloną populację: {pop} (dozwolone: {self.population_mapper.active_populations})")

        count = self.class_counts.get((pop, wiek), 0)
        desired_total = self.max_count
        augment_needed = max(0, desired_total - count)
        prob = min(1.0, augment_needed / desired_total)

        if torch.rand(1).item() < prob:
            self.augment_applied[(pop, wiek)] += 1
            transform = self.transform_strong

            if not self._printed_augmentation_notice:
                print("✨ Augmentacja w toku dla klas o małej liczności...")
                self._printed_augmentation_notice = True
        else:
            transform = self.transform_base

        label_idx = self.population_mapper.to_idx(pop)     # [ZAMIANA NAJWAŻNIEJSZA] indeks zamiast label

        return transform(image), label_idx, {"populacja": torch.tensor(pop), "wiek": torch.tensor(wiek)}
        # ^^^ [ZAMIANA] zwracamy tylko indeks klasy zgodny z model/training
