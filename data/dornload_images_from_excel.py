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

    required_columns = ["Populacja", "Typ otolitu"]
    column_candidates = ["path", "FilePath"]
    path_column = next((col for col in column_candidates if col in df.columns), None)

    for col in required_columns:
        if col not in df.columns:
            print(f"❌ Brak wymaganej kolumny '{col}' w pliku Excel.")
            return
    if not path_column:
        print("❌ Nie znaleziono kolumny z ścieżkami do zdjęć (szukano 'path' lub 'FilePath').")
        return
    if "Wiek" not in df.columns:
        print("❌ Kolumna 'Wiek' nie istnieje w pliku Excel.")
        return

    # Filtrowanie i reset indeksów
    df = df[[path_column, "Populacja", "Typ otolitu", "Wiek"]].dropna()
    df = df[df["Typ otolitu"].str.contains("Left|Right", case=False, na=False)]
    df["Populacja"] = df["Populacja"].astype(int)
    df = df[df["Populacja"].isin([1, 2])]
    df = df.reset_index(drop=True)

    print(f"🔍 Znaleziono {len(df)} spełniających warunki zdjęć...")

    copied_count = 0
    missing_count = 0

    for i, row in df.iterrows():
        raw_path = str(row[path_column]).strip()
        if not os.path.exists(raw_path):
            print(f"❌ [{i}] Plik nie istnieje: {raw_path}")
            missing_count += 1
            continue

        populacja = row["Populacja"]
        population_folder = str(populacja)

        source_path = Path(raw_path)
        target_dir = BASE_SAVE_DIR / population_folder
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_path.name

        try:
            shutil.copy2(source_path, target_path)
            print(f"✅ [{i}] Skopiowano do populacji {population_folder}: {source_path.name}")
            copied_count += 1
        except Exception as e:
            print(f"❌ [{i}] Błąd kopiowania {source_path}: {e}")

    print(f"\n🎉 Zakończono kopiowanie zdjęć.")
    print(f"✅ Skopiowano: {copied_count}")
    print(f"❗Brakujących plików: {missing_count}")

    # 🔢 Podsumowanie wieku
    for pop_val in [1, 2]:
        pop_df = df[df["Populacja"] == pop_val]
        if pop_df.empty:
            continue
        age_summary = pop_df["Wiek"].value_counts().sort_index()
        print(f"\n📊 Podsumowanie wieku dla populacji {pop_val}:")
        print(age_summary)

        summary_path = BASE_SAVE_DIR / f"age_summary_{pop_val}.csv"
        age_summary.to_csv(summary_path, header=["Liczba"], index_label="Wiek")
        print(f"💾 Zapisano podsumowanie do: {summary_path}")

if __name__ == "__main__":
    download_images_from_excel()
