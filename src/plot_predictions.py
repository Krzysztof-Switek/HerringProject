import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Lista modeli do analizy – można dodać więcej nazw
model_names = ["efficientnet_v2_l"]

def plot_correct_predictions_per_population(df: pd.DataFrame, output_dir: Path, model_name: str):
    populations = df["Populacja"].unique()

    for pop in populations:
        df_pop = df[df["Populacja"] == pop].copy()
        pred_col = f"{model_name}_pred"
        df_pop["correct"] = df_pop[pred_col] == df_pop["Populacja"]

        # Grupowanie danych
        correct_counts = df_pop[df_pop["correct"]].groupby("Wiek").size()
        incorrect_counts = df_pop[~df_pop["correct"]].groupby("Wiek").size()

        all_ages = sorted(df_pop["Wiek"].unique())
        correct_counts = correct_counts.reindex(all_ages, fill_value=0)
        incorrect_counts = incorrect_counts.reindex(all_ages, fill_value=0)

        total_counts = correct_counts + incorrect_counts
        accuracy = (correct_counts / total_counts).round(2)

        # Utwórz siatkę wykres + tabela
        fig = plt.figure(figsize=(10, 8))
        spec = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[5, 1])

        # Górny wykres
        ax = fig.add_subplot(spec[0])
        x = range(len(all_ages))
        bars_correct = ax.bar(x, correct_counts, color='green', label="Poprawne")
        bars_incorrect = ax.bar(x, incorrect_counts, bottom=correct_counts, color='red', label="Niepoprawne")

        ax.set_title(f"Model: {model_name} – Populacja {pop}\nPoprawne i niepoprawne predykcje vs wiek", fontsize=12)
        ax.set_xlabel("Wiek")
        ax.set_ylabel("Liczba predykcji")
        ax.set_xticks(x)
        ax.set_xticklabels([str(age) for age in all_ages])
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

        # Etykiety na słupkach
        for i, (bar1, bar2) in enumerate(zip(bars_correct, bars_incorrect)):
            correct = correct_counts.iloc[i]
            incorrect = incorrect_counts.iloc[i]

            # Poprawne (zielone)
            if correct > 0:
                height = bar1.get_height()
                if height >= 15:
                    y = bar1.get_y() + height / 2
                    color = "white"
                    va = "center"
                else:
                    y = bar1.get_y() + height + 1
                    color = "black"
                    va = "bottom"
                ax.text(
                    bar1.get_x() + bar1.get_width() / 2,
                    y,
                    str(correct),
                    ha='center',
                    va=va,
                    color=color,
                    fontsize=8,
                    weight='bold'
                )

            # Niepoprawne (czerwone)
            if incorrect > 0:
                height_incorrect = bar2.get_height()
                if height_incorrect >= 15:
                    y = bar2.get_y() + height_incorrect / 2
                    color = "white"
                    va = "center"
                else:
                    if correct > 0:
                        y = bar1.get_y() + bar1.get_height() + 24
                    else:
                        y = bar2.get_y() + height_incorrect + 1
                    color = "black"
                    va = "bottom"
                ax.text(
                    bar2.get_x() + bar2.get_width() / 2,
                    y,
                    str(incorrect),
                    ha='center',
                    va=va,
                    color=color,
                    fontsize=8,
                    weight='bold'
                )

        # Tabela na dole
        ax_table = fig.add_subplot(spec[1])
        ax_table.axis('off')  # wyłącz osie
        table_data = [[str(age), f"{acc:.2f}"] for age, acc in zip(all_ages, accuracy)]
        table_df = pd.DataFrame(table_data, columns=["Wiek", "Accuracy"])
        table = ax_table.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)  # zwiększa wysokość wierszy

        plt.tight_layout()

        # Zapis
        output_path = output_dir / f"{model_name}_stacked_predictions_populacja_{pop}.png"
        plt.savefig(output_path)
        print(f"✅ Zapisano: {output_path}")
        plt.close()


def main():
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        excel_path = Path(f"data_loader/AnalysisWithOtolithPhoto_with_sets_with_preds_{model_name}.xlsx")
        df = pd.read_excel(excel_path)

        # Filtrowanie tylko jeśli SET niepusty
        df = df[df["SET"].notna()]

        # Konwersje typów
        df["Populacja"] = df["Populacja"].astype(int)
        pred_col = f"{model_name}_pred"
        df[pred_col] = df[pred_col].astype("Int64")
        df["Wiek"] = df["Wiek"].astype(int)

        plot_correct_predictions_per_population(df, output_dir, model_name)


if __name__ == "__main__":
    main()
