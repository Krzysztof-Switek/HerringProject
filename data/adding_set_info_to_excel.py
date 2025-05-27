import pandas as pd
from pathlib import Path

def extract_filename(path: str) -> str:
    """Zwraca samą nazwę pliku z pełnej ścieżki"""
    return Path(path).name.lower()

def extract_filename_from_txt(line: str) -> str:
    """Zwraca nazwę pliku z linii typu 1\image.jpg lub 2\image.jpg"""
    return line.strip().split("\\")[-1].lower()

def load_file_list(path: Path, set_name: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        filenames = [extract_filename_from_txt(line) for line in f.readlines()]
    return pd.DataFrame({"FileName": filenames, "SET": set_name})

def main():
    # Ścieżki
    excel_path = Path("C:/Users/kswitek/Documents/HerringProject/src/data_loader/AnalysisWithOtolithPhoto.xlsx")
    train_txt = Path("C:/Users/kswitek/Documents/HerringProject/data/train/train_files.txt")
    val_txt = Path("C:/Users/kswitek/Documents/HerringProject/data/val/val_files.txt")
    test_txt = Path("C:/Users/kswitek/Documents/HerringProject/data/test/test_files.txt")

    # Wczytaj Excel i dodaj FileName
    df_excel = pd.read_excel(excel_path)
    if "FilePath" not in df_excel.columns:
        raise ValueError("Brak kolumny 'FilePath' w pliku Excel.")

    df_excel["FileName"] = df_excel["FilePath"].apply(extract_filename)

    # Wczytaj listy plików z SET-em
    train_df = load_file_list(train_txt, "TRAIN")
    val_df = load_file_list(val_txt, "VAL")
    test_df = load_file_list(test_txt, "TEST")

    all_sets_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Dopasowanie po FileName
    merged_df = pd.merge(df_excel, all_sets_df, how="left", on="FileName")

    # Zapisz nowy Excel
    output_path = excel_path.parent / "AnalysisWithOtolithPhoto_with_sets.xlsx"
    merged_df.to_excel(output_path, index=False)

    print(f"Nowy plik zapisany: {output_path}")

if __name__ == "__main__":
    main()
