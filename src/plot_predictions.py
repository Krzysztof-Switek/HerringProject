import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_correct_predictions_per_population(df: pd.DataFrame, output_dir: Path):
    populations = df["Populacja"].unique()

    for pop in populations:
        df_pop = df[df["Populacja"] == pop].copy()

        # Czy predykcja poprawna
        df_pop["correct"] = df_pop["resnet50_pred"] == df_pop["Populacja"]

        # Liczba trafień i wszystkich na wiek
        correct_counts = df_pop[df_pop["correct"]].groupby("Wiek").size()
        total_counts = df_pop.groupby("Wiek").size()
        correct_counts = correct_counts.reindex(total_counts.index, fill_value=0)

        # Accuracy per age
        accuracy = (correct_counts / total_counts).round(2)

        # Tworzenie wykresu i tabeli
        fig, ax = plt.subplots(figsize=(8, 6))

        correct_counts.plot(kind="bar", color="green", ax=ax)
        ax.set_title(f"Populacja {pop} – poprawne predykcje vs wiek")
        ax.set_xlabel("Wiek")
        ax.set_ylabel("Liczba poprawnych predykcji")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xticklabels(correct_counts.index.astype(str), rotation=0)
        ax.legend(["Poprawne predykcje"])

        # Dodanie tabeli z accuracy
        table_data = [[str(age), str(acc)] for age, acc in zip(accuracy.index, accuracy)]
        table = plt.table(
            cellText=table_data,
            colLabels=["Wiek", "Accuracy"],
            cellLoc="center",
            colLoc="center",
            loc="bottom",
            bbox=[0.0, -0.35, 1.0, 0.25]  # [left, bottom, width, height]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        plt.subplots_adjust(bottom=0.3)
        plt.tight_layout()

        plt.show()

        # Zapis
        output_path = output_dir / f"correct_predictions_populacja_{pop}.png"
        fig.savefig(output_path)
        print(f"✅ Zapisano: {output_path}")
        plt.close()


def main():
    excel_path = Path("src/data_loader/AnalysisWithOtolithPhoto_with_sets.xlsx")
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Wczytaj dane
    df = pd.read_excel(excel_path)

    # Filtrowanie tylko jeśli SET niepusty
    df = df[df["SET"].notna()]

    # Konwersje typów
    df["Populacja"] = df["Populacja"].astype(int)
    df["resnet50_pred"] = df["resnet50_pred"].astype("Int64")
    df["Wiek"] = df["Wiek"].astype(int)

    plot_correct_predictions_per_population(df, output_dir)


if __name__ == "__main__":
    main()
