"""
Skrypt do analizy wyników predykcji modelu.
"""

import pandas as pd
from pathlib import Path
import glob

# ===================================================================
# 🔧 USTAWIENIA UŻYTKOWNIKA
# ===================================================================

# 1. Podaj ścieżkę do katalogu z logami, który zawiera plik .xlsx z predykcjami.
LOG_DIR = "C:/Users/kswitek/Documents/HerringProject/results/logs/BEST_resnet50_standard_ce_multi_2025-07-19_19-24"

# 2. Zdefiniuj mapowanie nazw kolumn na ich adresy w pliku Excel.
#    Nazwy po lewej stronie są używane w skrypcie. Adresy po prawej to kolumny w Excelu.
COLUMN_MAPPING = {
    # Dane biologiczne
    "Wiek": "U",
    "Populacja": "AH",
    # Dane z modelu
    "SET": "AQ",
    "standard_ce_pred": "AR",
    "standard_ce_prob": "AS",
    "standard_ce_age_pred": "AT",
}
# ===================================================================

def excel_col_to_int(col_str: str) -> int:
    """Konwertuje nazwę kolumny Excela (np. 'A', 'AR') na indeks liczbowy (zaczynając od 0)."""
    index = 0
    for char in col_str:
        index = index * 26 + (ord(char.upper()) - ord('A')) + 1
    return index - 1

def main():
    """Główna funkcja skryptu."""
    log_dir = Path(LOG_DIR)

    if not log_dir.is_dir():
        print(f"Błąd: Podana ścieżka '{log_dir}' nie jest prawidłowym katalogiem.")
        return

    xlsx_files = glob.glob(str(log_dir / "*.xlsx"))
    if not xlsx_files:
        print(f"Błąd: Nie znaleziono pliku .xlsx w katalogu '{log_dir}'.")
        return

    predictions_path = Path(xlsx_files[0])
    print(f"Znaleziono plik z predykcjami: {predictions_path.name}")

    try:
        # Przygotuj listę kolumn do wczytania i mapowanie do nowych nazw
        col_indices = [excel_col_to_int(v) for v in COLUMN_MAPPING.values()]
        col_names = list(COLUMN_MAPPING.keys())

        # Wczytaj tylko określone kolumny
        df = pd.read_excel(
            predictions_path,
            engine='openpyxl',
            header=None,  # Wczytaj bez nagłówka, bo nazwiemy je sami
            usecols=col_indices
        )

        # Nadaj kolumnom nowe, logiczne nazwy
        # Upewnij się, że kolejność jest prawidłowa
        map_to_new_names = {df.columns[i]: col_names[i] for i in range(len(col_names))}
        df.rename(columns=map_to_new_names, inplace=True)


        # Ignoruj wiersze, gdzie 'SET' jest pusty
        df_filtered = df[df['SET'].notna()].copy()

        if df_filtered.empty:
            print("Błąd: Brak danych do analizy po odfiltrowaniu pustych wartości w kolumnie 'SET'.")
            return

        print("\nPrzetworzone dane (pierwsze 5 wierszy):")
        print(df_filtered.head().to_string())

    except Exception as e:
        print(f"Wystąpił błąd podczas przetwarzania pliku: {e}")


if __name__ == "__main__":
    main()
