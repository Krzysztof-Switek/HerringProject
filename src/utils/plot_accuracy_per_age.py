import pandas as pd
import matplotlib.pyplot as plt
import itertools

EXCEL_PATH = "C:/Users/kswitek/Documents/HerringProject/src/data_loader/all_predictions_2025-06-14.xlsx"

LOSS_TYPES = [
    "standard_ce",
    "sample_weighted_ce",
    "weighted_age_ce",
    "focal_loss_ageboost",
    "ldam",
    "class_balanced_focal",
    "focal_tversky",
    "ghm",
    "seesaw",
    "asymmetric_focal"
]

COLOR_CYCLE = itertools.cycle([
    '#e6194b',  # czerwony
    '#3cb44b',  # zielony
    '#ffe119',  # żółty
    '#4363d8',  # niebieski
    '#f58231',  # pomarańczowy
    '#911eb4',  # fioletowy
    '#46f0f0',  # turkusowy
    '#f032e6',  # różowy
    '#bcf60c',  # limonkowy
    '#fabebe'   # jasnoróżowy
])


def plot_accuracy_per_age(populacja_id):
    try:
        df = pd.read_excel(EXCEL_PATH)
    except FileNotFoundError:
        print(f"❌ Nie znaleziono pliku {EXCEL_PATH}")
        return

    if "Populacja" not in df.columns or "Wiek" not in df.columns:
        print("❌ Brakuje kolumn 'Populacja' lub 'Wiek'")
        return

    df = df[(df["Populacja"] == populacja_id) & (df["Wiek"] != -9)]

    plt.figure(figsize=(10, 6))
    for loss in LOSS_TYPES:
        pred_col = f"{loss}_pred"
        if pred_col not in df.columns:
            print(f"⚠️ Kolumna {pred_col} nie istnieje – pomijam")
            continue

        df_loss = df.dropna(subset=[pred_col]).copy()
        df_loss[pred_col] = df_loss[pred_col].astype(int)

        grouped = (
            df_loss.groupby("Wiek")[["Populacja", pred_col]]
            .apply(lambda x: (x[pred_col] == x["Populacja"]).sum() / len(x))
            .sort_index()
        )

        if grouped.empty:
            print(f"⚠️ Brak danych dla funkcji straty: {loss}")
            continue

        color = next(COLOR_CYCLE)
        plt.plot(grouped.index, grouped.values, label=loss, linewidth=2, color=color)

    all_wieki = sorted(df["Wiek"].unique())
    plt.xticks(all_wieki)
    plt.axhline(y=0.75, color='gray', linestyle='--', linewidth=1)
    plt.text(all_wieki[0], 0.752, "poziom 0.75", color='gray')

    plt.title(f"Accuracy wg wieku – populacja {populacja_id}")
    plt.xlabel("Wiek")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_sample_counts_per_age():
    try:
        df = pd.read_excel(EXCEL_PATH)
    except FileNotFoundError:
        print(f"❌ Nie znaleziono pliku {EXCEL_PATH}")
        return

    if "Populacja" not in df.columns or "Wiek" not in df.columns:
        print("❌ Brakuje kolumn 'Populacja' lub 'Wiek'")
        return

    # ✅ Utrzymujemy tylko populacje 1 i 2
    df = df[(df["Wiek"] != -9) & (df["Populacja"].isin([1, 2]))]
    grouped = df.groupby(["Wiek", "Populacja"]).size().unstack(fill_value=0)

    ax = grouped.plot(kind="bar", figsize=(12, 6), width=0.8)
    plt.title("Liczba próbek w każdej klasie wieku (populacje 1 i 2)")
    plt.xlabel("Wiek")
    plt.ylabel("Liczba przypadków")
    plt.xticks(rotation=0)
    plt.grid(True, axis='y')
    plt.legend(title="Populacja")

    for i, age in enumerate(grouped.index):
        for j, pop in enumerate(grouped.columns):
            value = grouped.loc[age, pop]
            ax.text(i + j * 0.25 - 0.15, value + 0.5, str(value), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()


def main():
    print("\n📊 Generuję wykres liczebności klas wieku dla obu populacji...")
    plot_sample_counts_per_age()

    print("\n📊 Generuję wykres dla populacji 1...")
    plot_accuracy_per_age(1)

    print("\n📊 Generuję wykres dla populacji 2...")
    plot_accuracy_per_age(2)


if __name__ == "__main__":
    main()
