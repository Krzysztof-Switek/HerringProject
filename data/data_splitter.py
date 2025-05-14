import os
import random
import shutil
from tqdm import tqdm

class DataSplitter:
    def __init__(self, base_dir=".", train_ratio=0.7, val_ratio=0.2, seed=42):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.seed = seed

        total = train_ratio + val_ratio + self.test_ratio
        if not (0.999 <= total <= 1.001):
            raise ValueError("Suma proporcji musi wynosić 1.0")

        random.seed(seed)

        self.split_files = {"train": [], "val": [], "test": []}

    def split_data(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Katalog {self.data_dir} nie istnieje")

        class_dirs = [d for d in os.listdir(self.data_dir)
                      if os.path.isdir(os.path.join(self.data_dir, d))
                      and d not in ["train", "val", "test"]]

        if not class_dirs:
            raise ValueError(f"Brak podkatalogów klas w {self.data_dir}")

        self._create_split_dirs()

        for class_dir in class_dirs:
            self._process_class(class_dir)

        self._save_file_lists()

    def _create_split_dirs(self):
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(self.data_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            for class_dir in os.listdir(self.data_dir):
                full_path = os.path.join(self.data_dir, class_dir)
                if os.path.isdir(full_path) and class_dir not in ["train", "val", "test"]:
                    os.makedirs(os.path.join(split_dir, class_dir), exist_ok=True)

    def _process_class(self, class_dir):
        src_dir = os.path.join(self.data_dir, class_dir)
        files = [f for f in os.listdir(src_dir)
                 if os.path.isfile(os.path.join(src_dir, f)) and not f.startswith('.')]

        if not files:
            print(f"Uwaga: Brak plików w {src_dir}")
            return

        random.shuffle(files)
        n = len(files)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        self._copy_files(train_files, class_dir, "train")
        self._copy_files(val_files, class_dir, "val")
        self._copy_files(test_files, class_dir, "test")

    def _copy_files(self, files, class_dir, split_name):
        src_dir = os.path.join(self.data_dir, class_dir)
        dst_dir = os.path.join(self.data_dir, split_name, class_dir)

        for file in tqdm(files, desc=f"Kopiowanie {split_name}/{class_dir}"):
            src = os.path.join(src_dir, file)
            dst = os.path.join(dst_dir, file)
            shutil.copy2(src, dst)
            self.split_files[split_name].append(os.path.join(class_dir, file))

    def _save_file_lists(self):
        for split, files in self.split_files.items():
            list_path = os.path.join(self.data_dir, split, f"{split}_files.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                for file in sorted(files):
                    f.write(file + "\n")
            print(f"📝 Zapisano listę plików do {list_path}")

if __name__ == "__main__":
    splitter = DataSplitter(
        base_dir="..",
        train_ratio=0.7,
        val_ratio=0.2,
        seed=42
    )

    print("🔄 Rozpoczynanie podziału danych w katalogu data/...")
    splitter.split_data()

    print("\n✅ Podział zakończony pomyślnie!")
    print("📂 Ostateczna struktura katalogów:")
    print("""
    data/
    ├── 1/              # Oryginalne zdjęcia klasy 0
    ├── 2/              # Oryginalne zdjęcia klasy 1
    ├── train/
    │   ├── 1/
    │   └── 2/
    │   └── train_files.txt
    ├── val/
    │   ├── 1/
    │   └── 2/
    │   └── val_files.txt
    └── test/
        ├── 1/
        └── 2/
        └── test_files.txt
    """)
