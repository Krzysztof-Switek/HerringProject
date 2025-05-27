import pandas as pd
from pathlib import Path
import shutil
import os

def cleanup_dataset_folders(excel_path: str, data_root: str):
    # Wczytaj dane z Excela
    df = pd.read_excel(excel_path)
    if "FilePath" not in df.columns:
        raise ValueError("Brak kolumny 'FilePath' w pliku Excel")

    # Wyciągnij same nazwy plików z pełnych ścieżek UNC
    valid_filenames = set(Path(p).name.lower() for p in df["FilePath"])
    print(f"📄 Wczytano {len(valid_filenames)} nazw plików z Excela.")

    # Foldery do przeszukania
    subfolders = ["train/0", "train/1", "val/0", "val/1", "test/0", "test/1"]
    data_root_path = Path(data_root)
    missing_files = valid_filenames.copy()
    matched_files = set()

    for sub in subfolders:
        folder = data_root_path / sub
        if not folder.exists():
            print(f"⚠️  Katalog nie istnieje: {folder}")
            continue

        to_delete_dir = folder / "do_usuniecia"
        to_delete_dir.mkdir(exist_ok=True)

        print(f"🔍 Sprawdzam katalog: {folder}")
        all_files = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.JPG"))

        for file in all_files:
            fname = file.name.lower()
            if fname in valid_filenames:
                matched_files.add(fname)
                missing_files.discard(fname)
            else:
                print(f"🚫 Plik NIE występuje w Excelu: {fname}")
                target_path = to_delete_dir / file.name
                shutil.move(str(file), str(target_path))
                print(f"🗂 Przeniesiono do: {target_path}")

    # Diagnostyka
    print(f"\n✅ Dopasowano {len(matched_files)} plików z listy.")
    print(f"❗ Nie dopasowano {len(missing_files)} plików z Excela do folderów.")
    if missing_files:
        print("🔸 Przykładowe brakujące:")
        for m in list(missing_files)[:10]:
            print(f"- {m}")

if __name__ == "__main__":
    excel_path = r"C:\Users\kswitek\Documents\HerringProject\src\data_loader\AnalysisWithOtolithPhoto.xlsx"
    data_root = r"C:\Users\kswitek\Documents\HerringProject\data"
    cleanup_dataset_folders(excel_path, data_root)
