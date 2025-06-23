import os
import torch


class AugmentWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, metadata, class_counts, max_count, transform_base, transform_strong, augment_applied):
        self.base_dataset = base_dataset
        self.metadata = metadata
        self.class_counts = class_counts
        self.max_count = max_count
        self.transform_base = transform_base
        self.transform_strong = transform_strong
        self.augment_applied = augment_applied

        self._printed_augmentation_notice = False

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        path, label = self.base_dataset.samples[index]
        image = self.base_dataset.loader(path)

        fname = os.path.basename(path).strip().lower()

        if fname not in self.metadata:
            print(f"⚠️ Nie znaleziono metadanych dla pliku: {fname}")
            transform = self.transform_base
            return transform(image), label, {"populacja": torch.tensor(-1), "wiek": torch.tensor(-1)}

        pop, wiek = self.metadata[fname]
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

        return transform(image), label, {"populacja": torch.tensor(pop), "wiek": torch.tensor(wiek)}
