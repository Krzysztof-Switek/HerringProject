import os
import random
import shutil
from tqdm import tqdm


class DataSplitter:
    def __init__(self, base_dir=".", train_ratio=0.7, val_ratio=0.2, seed=42):
        """
        Inicjalizacja splittera danych działającego w katalogu data/

        Args:
            base_dir: Główny katalog projektu (domyślnie ".")
            train_ratio: Proporcja danych treningowych (0-1)
            val_ratio: Proporcja danych walidacyjnych (0-1)
            seed: Ziarno losowości dla powtarzalności
        """
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.seed = seed

        # Walidacja proporcji
        total = train_ratio + val_ratio + self.test_ratio
        if not (0.999 <= total <= 1.001):
            raise ValueError("Suma proporcji musi wynosić 1.0")

        random.seed(seed)

    def split_data(self):
        """
        Główna metoda wykonująca podział danych wewnątrz katalogu data/
        """
        # Sprawdzenie struktury wejściowej
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Katalog {self.data_dir} nie istnieje")

        class_dirs = [d for d in os.listdir(self.data_dir)
                      if os.path.isdir(os.path.join(self.data_dir, d))]
        if not class_dirs:
            raise ValueError(f"Brak podkatalogów klas w {self.data_dir}")

        # Tworzenie podkatalogów train/val/test
        self._create_split_dirs()

        # Przetwarzanie każdej klasy
        for class_dir in class_dirs:
            self._process_class(class_dir)

    def _create_split_dirs(self):
        """Tworzy podkatalogi train/val/test w data/"""
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(self.data_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            # Tworzenie podkatalogów klas w każdym splicie
            for class_dir in os.listdir(self.data_dir):
                if os.path.isdir(os.path.join(self.data_dir, class_dir)) and class_dir not in ["train", "val", "test"]:
                    os.makedirs(
                        os.path.join(split_dir, class_dir),
                        exist_ok=True
                    )

    def _process_class(self, class_dir):
        """
        Przetwarza pojedynczą klasę zdjęć
        """
        src_dir = os.path.join(self.data_dir, class_dir)

        # Pomijanie istniejących katalogów splitów
        if class_dir in ["train", "val", "test"]:
            return

        files = [f for f in os.listdir(src_dir)
                 if os.path.isfile(os.path.join(src_dir, f)) and not f.startswith('.')]

        if not files:
            print(f"Uwaga: Brak plików w {src_dir}")
            return

        # Losowy podział plików
        random.shuffle(files)
        n = len(files)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        # Kopiowanie plików z paskiem postępu
        self._copy_files(train_files, class_dir, "train")
        self._copy_files(val_files, class_dir, "val")
        self._copy_files(test_files, class_dir, "test")

    def _copy_files(self, files, class_dir, split_name):
        """
        Kopiuje pliki do odpowiedniego podkatalogu w data/
        """
        src_dir = os.path.join(self.data_dir, class_dir)
        dst_dir = os.path.join(self.data_dir, split_name, class_dir)

        for file in tqdm(files, desc=f"Kopiowanie {split_name}/{class_dir}"):
            src = os.path.join(src_dir, file)
            dst = os.path.join(dst_dir, file)
            shutil.copy2(src, dst)


if __name__ == "__main__":
    # Konfiguracja
    splitter = DataSplitter(
        base_dir="..",  # Wskazuje, że data/ jest w katalogu nadrzędnym
        train_ratio=0.7,  # 70% treningowe
        val_ratio=0.2,  # 20% walidacyjne
        seed=42  # Ziarno losowości
    )

    # Uruchomienie podziału
    print("Rozpoczynanie podziału danych w katalogu data/...")
    splitter.split_data()

    print("\nPodział zakończony pomyślnie!")
    print("Ostateczna struktura katalogu data/:")
    print("""
    data/
    ├── 1/              # Oryginalne zdjęcia klasy 1
    ├── 2/              # Oryginalne zdjęcia klasy 2
    ├── train/
    │   ├── 1/          # Zdjęcia treningowe klasy 1
    │   └── 2/          # Zdjęcia treningowe klasy 2
    ├── val/
    │   ├── 1/          # Zdjęcia walidacyjne klasy 1
    │   └── 2/          # Zdjęcia walidacyjne klasy 2
    └── test/
        ├── 1/          # Zdjęcia testowe klasy 1 (model nigdy nie widzi)
        └── 2/          # Zdjęcia testowe klasy 2 (model nigdy nie widzi)
    """)