import pandas as pd
import random
from pathlib import Path

def resolve_kumulacyjne(predictions: list) -> int:
    # Usuwamy wartości brakujące
    filtered = [int(p) for p in predictions if pd.notna(p)]

    if len(filtered) < 3:
        return pd.NA  # Zbyt mało danych do sensownego głosowania

    counts = {1: filtered.count(1), 2: filtered.count(2)}

    # Głosowanie większościowe — poprawna logika
    if counts[1] > counts[2]:
        return 1
    elif counts[2] > counts[1]:
        return 2
    else:
        return random.choice([1, 2])


def main():
    excel_path = Path("data_loader/AnalysisWithOtolithPhoto_with_sets_ALL_4_MODELS.xlsx")
    df = pd.read_excel(excel_path)

    base_columns = ["FileName", "Populacja", "Wiek", "SET"]
    model_columns = [
        "convnext_large_pred", "convnext_large_probability",
        "efficientnet_v2_l_pred", "efficientnet_v2_l_probability",
        "resnet50_pred", "resnet50_probability",
        "regnet_y_32gf_pred", "regnet_y_32gf_probability"

    ]
    pred_columns = [col for col in model_columns if col.endswith("_pred")]

    df_selected = df[base_columns + model_columns].copy()

    # Usuwamy wiersze, w których Populacja, Wiek lub SET są puste
    df_selected.dropna(subset=["Populacja", "Wiek", "SET"], inplace=True)

    df_selected["Populacja"] = df_selected["Populacja"].astype(int)
    df_selected["Wiek"] = df_selected["Wiek"].astype(int)

    for col in pred_columns:
        df_selected[col] = df_selected[col].astype("Int64")

    # Kolumna PRED_KUMULACYJNE
    df_selected["PRED_KUMULACYJNE"] = df_selected.apply(
        lambda row: resolve_kumulacyjne([row[col] for col in pred_columns]), axis=1
    )

    # Podgląd
    print(df_selected[base_columns + pred_columns + ["PRED_KUMULACYJNE"]].head())

    # Zapis
    output_path = Path("data_loader/AnalysisWithOtolithPhoto_with_preds_KUMULACYJNE.xlsx")
    df_selected.to_excel(output_path, index=False)
    print(f"✅ Zapisano plik: {output_path}")

if __name__ == "__main__":
    main()
