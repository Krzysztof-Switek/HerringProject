import os
import random
import shutil
from tqdm import tqdm

class DataSplitter:
    def __init__(self, base_dir=".", source_subdir="server_data", train_ratio=0.7, val_ratio=0.2, seed=42):
        self.base_dir = base_dir
        self.source_dir = os.path.join(base_dir, "data", source_subdir)
        self.target_dir = os.path.join(base_dir, "data")
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
        if not os.path.exists(self.source_dir):
            raise FileNotFoundError(f"Katalog {self.source_dir} nie istnieje")

        class_dirs = [d for d in os.listdir(self.source_dir)
                      if os.path.isdir(os.path.join(self.source_dir, d))]

        if not class_dirs:
            raise ValueError(f"Brak podkatalogów klas w {self.source_dir}")

        self._create_split_dirs(class_dirs)

        for class_dir in class_dirs:
            self._process_class(class_dir)

        self._save_file_lists()

    def _create_split_dirs(self, class_dirs):
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(self.target_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            for class_dir in class_dirs:
                os.makedirs(os.path.join(split_dir, class_dir), exist_ok=True)

    def _process_class(self, class_dir):
        src_dir = os.path.join(self.source_dir, class_dir)
        files = [f for f in os.listdir(src_dir)
                 if os.path.isfile(os.path.join(src_dir, f)) and not f.startswith('.')]

        if not files:
            print(f"⚠️ Brak plików w {src_dir}")
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
        src_dir = os.path.join(self.source_dir, class_dir)
        dst_dir = os.path.join(self.target_dir, split_name, class_dir)

        for file in tqdm(files, desc=f"Kopiowanie {split_name}/{class_dir}"):
            src = os.path.join(src_dir, file)
            dst = os.path.join(dst_dir, file)
            shutil.copy2(src, dst)
            self.split_files[split_name].append(os.path.join(class_dir, file))

    def _save_file_lists(self):
        for split, files in self.split_files.items():
            list_path = os.path.join(self.target_dir, split, f"{split}_files.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                for file in sorted(files):
                    f.write(file + "\n")
            print(f"📝 Zapisano listę plików do {list_path}")

if __name__ == "__main__":
    splitter = DataSplitter(
        base_dir="C:/Users/kswitek/Documents/HerringProject",
        source_subdir="server_data",
        train_ratio=0.7,
        val_ratio=0.2,
        seed=42
    )

    print("🔄 Rozpoczynanie podziału danych z katalogu data/server_data/")
    splitter.split_data()

    print("\n✅ Podział zakończony pomyślnie!")
    print("📂 Ostateczna struktura katalogów:")
    print("""
    data/
    ├── train/
    │   ├── 0/
    │   ├── 1/
    │   └── train_files.txt
    ├── val/
    │   ├── 0/
    │   ├── 1/
    │   └── val_files.txt
    ├── test/
    │   ├── 0/
    │   ├── 1/
    │   └── test_files.txt
    """)
