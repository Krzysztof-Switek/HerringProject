import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# 🔧 ZDEFINIUJ ŚCIEŻKĘ DO PLIKU .XLSX NA POCZĄTKU
from pathlib import Path

excel_path = Path(r"C:\Users\kswitek\Documents\HerringProject\results\logs\efficientnet_v2_l_focal_tversky_15-06_08-33\all_predictions_2025-06-15.xlsx")

output_dir = excel_path.parent
loss_name = "focal_tversky"  # musi pasować do kolumn: focal_tversky_pred, focal_tversky_prob

def plot_correct_predictions_per_population(df: pd.DataFrame, output_dir: Path, loss_name: str):
    populations = df["Populacja"].unique()

    for pop in populations:
        df_pop = df[df["Populacja"] == pop].copy()
        pred_col = f"{loss_name}_pred"
        df_pop["correct"] = df_pop[pred_col] == df_pop["Populacja"]

        correct_counts = df_pop[df_pop["correct"]].groupby("Wiek").size()
        incorrect_counts = df_pop[~df_pop["correct"]].groupby("Wiek").size()

        all_ages = sorted(df_pop["Wiek"].unique())
        correct_counts = correct_counts.reindex(all_ages, fill_value=0)
        incorrect_counts = incorrect_counts.reindex(all_ages, fill_value=0)

        total_counts = correct_counts + incorrect_counts
        accuracy = (correct_counts / total_counts).round(2)

        fig = plt.figure(figsize=(10, 8))
        spec = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[5, 1])
        ax = fig.add_subplot(spec[0])
        x = range(len(all_ages))

        bars_correct = ax.bar(x, correct_counts, color='green', label="Poprawne")
        bars_incorrect = ax.bar(x, incorrect_counts, bottom=correct_counts, color='red', label="Niepoprawne")

        ax.set_title(f"Funkcja straty: {loss_name} – Populacja {pop}\nPoprawne i niepoprawne predykcje vs wiek", fontsize=12)
        ax.set_xlabel("Wiek")
        ax.set_ylabel("Liczba predykcji")
        ax.set_xticks(x)
        ax.set_xticklabels([str(age) for age in all_ages])
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

        for i, (bar1, bar2) in enumerate(zip(bars_correct, bars_incorrect)):
            correct = correct_counts.iloc[i]
            incorrect = incorrect_counts.iloc[i]

            if correct > 0:
                height = bar1.get_height()
                y = bar1.get_y() + height / 2 if height >= 15 else bar1.get_y() + height + 1
                color = "white" if height >= 15 else "black"
                va = "center" if height >= 15 else "bottom"
                ax.text(bar1.get_x() + bar1.get_width() / 2, y, str(correct), ha='center', va=va, color=color, fontsize=8, weight='bold')

            if incorrect > 0:
                height_incorrect = bar2.get_height()
                if height_incorrect >= 15:
                    y = bar2.get_y() + height_incorrect / 2
                    color = "white"
                    va = "center"
                else:
                    y = bar1.get_y() + bar1.get_height() + 24 if correct > 0 else bar2.get_y() + height_incorrect + 1
                    color = "black"
                    va = "bottom"
                ax.text(bar2.get_x() + bar2.get_width() / 2, y, str(incorrect), ha='center', va=va, color=color, fontsize=8, weight='bold')

        ax_table = fig.add_subplot(spec[1])
        ax_table.axis('off')
        table_data = [[str(age), f"{acc:.2f}"] for age, acc in zip(all_ages, accuracy)]
        table_df = pd.DataFrame(table_data, columns=["Wiek", "Accuracy"])
        table = ax_table.table(cellText=table_df.values, colLabels=table_df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)

        plt.tight_layout()

        output_path = output_dir / f"{loss_name}_stacked_predictions_populacja_{pop}.png"
        plt.savefig(output_path)
        print(f"✅ Zapisano: {output_path}")
        plt.close()


def main():
    df = pd.read_excel(excel_path)
    df = df[df["SET"].notna()]
    df["Populacja"] = df["Populacja"].astype(int)
    df[f"{loss_name}_pred"] = df[f"{loss_name}_pred"].astype("Int64")
    df["Wiek"] = df["Wiek"].astype(int)

    plot_correct_predictions_per_population(df, output_dir, loss_name)

if __name__ == "__main__":
    main()
