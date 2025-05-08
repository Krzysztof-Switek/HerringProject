import os
import shutil
import pandas as pd
from pathlib import Path

EXCEL_PATH = Path(r"C:\Users\kswitek\Documents\HerringProject\src\data_loader\AnalysisWithOtolithPhoto.xlsx")
BASE_SAVE_DIR = Path("server_data")

def download_images_from_excel():
    try:
        df = pd.read_excel(EXCEL_PATH, engine='openpyxl')
    except Exception as e:
        print(f"❌ Błąd wczytywania pliku Excel: {e}")
        return

    # Sprawdź, czy kolumny istnieją
    if "Populacja" not in df.columns:
        print("❌ Brak kolumny 'Populacja' w pliku Excel.")
        return

    column_candidates = ["path", "FilePath"]
    path_column = next((col for col in column_candidates if col in df.columns), None)

    if not path_column:
        print("❌ Nie znaleziono kolumny z ścieżkami do zdjęć (szukano 'path' lub 'FilePath').")
        return

    df = df[[path_column, "Populacja"]].dropna()

    print(f"🔍 Znaleziono {len(df)} wierszy z przypisaną populacją i ścieżką...")

    for i, row in df.iterrows():
        source_path = Path(row[path_column])
        population = str(int(row["Populacja"]))  # np. "1" lub "2"
        target_dir = BASE_SAVE_DIR / population
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_path.name

        try:
            if not source_path.exists():
                print(f"❌ [{i}] Plik nie istnieje: {source_path}")
                continue

            shutil.copy2(source_path, target_path)
            print(f"✅ [{i}] Skopiowano do populacji {population}: {source_path.name}")
        except Exception as e:
            print(f"❌ [{i}] Błąd kopiowania {source_path}: {e}")

    print("🎉 Zakończono kopiowanie zdjęć.")

if __name__ == "__main__":
    download_images_from_excel()
