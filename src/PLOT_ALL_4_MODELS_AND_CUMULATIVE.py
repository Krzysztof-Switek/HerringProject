import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image

model_to_column_map = {
    "efficientnet_v2_l": {
        "pred": "efficientnet_v2_l_pred",
        "prob": "efficientnet_v2_l_probability"
    },
    "resnet50": {
        "pred": "resnet50_pred",
        "prob": "resnet50_probability"
    },
    "regnet_y_32gf": {
        "pred": "regnet_y_32gf_pred",
        "prob": "regnet_y_32gf_probability"
    },
    "convnext_large": {
        "pred": "convnext_large_pred",
        "prob": "convnext_large_probability"
    },
    "PRED_KUMULACYJNE": {
        "pred": "PRED_KUMULACYJNE",
        "prob": None
    }
}

def plot_correct_predictions_per_population(df: pd.DataFrame, output_dir: Path, model_name: str, pred_col: str, show_table: bool):
    populations = df["Populacja"].unique()
    for pop in populations:
        df_pop = df[df["Populacja"] == pop].copy()
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

        ax.set_title(f"Model: {model_name} – Populacja {pop}\nPoprawne i niepoprawne predykcje vs wiek", fontsize=12)
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
                y = bar1.get_y() + (height / 2 if height >= 15 else height + 1)
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

        if show_table:
            ax_table = fig.add_subplot(spec[1])
            ax_table.axis('off')
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
            table.scale(1, 1.6)

        plt.tight_layout()
        filename = f"{model_name}_stacked_predictions_populacja_{pop}.png"
        plt.savefig(output_dir / filename)
        print(f"✅ Zapisano: {filename}")
        plt.close()

def plot_accuracy_line_chart(df: pd.DataFrame, output_dir: Path):
    models_to_plot = [k for k in model_to_column_map if k != "PRED_KUMULACYJNE"]
    for pop in [1, 2]:
        df_pop = df[df["Populacja"] == pop].copy()
        age_classes = sorted(df_pop["Wiek"].unique())
        model_accuracies = {}
        for model_name in models_to_plot:
            pred_col = model_to_column_map[model_name]["pred"]
            df_pop["correct"] = df_pop[pred_col] == df_pop["Populacja"]
            accuracy_per_age = (
                df_pop.groupby("Wiek")["correct"].mean().reindex(age_classes, fill_value=0).round(2)
            )
            model_accuracies[model_name] = accuracy_per_age

        plt.figure(figsize=(10, 6))
        for model_name, acc_series in model_accuracies.items():
            plt.plot(age_classes, acc_series.values, marker='o', label=model_name)

        plt.title(f"Accuracy per age class – Populacja {pop}")
        plt.xlabel("Wiek")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(title="Model")
        plt.xticks(age_classes)
        plt.tight_layout()

        output_path = output_dir / f"accuracy_per_age_populacja_{pop}.png"
        plt.savefig(output_path)
        print(f"📈 Zapisano wykres liniowy dla populacji {pop}: {output_path}")
        plt.close()

def save_all_plots_to_pdf(images_dir: Path, output_pdf_path: Path):
    png_files = sorted(images_dir.glob("*.png"))
    if not png_files:
        print("⚠️ Brak plików PNG do połączenia.")
        return
    images = [Image.open(png).convert("RGB") for png in png_files]
    images[0].save(output_pdf_path, save_all=True, append_images=images[1:])
    print(f"📄 Zapisano zbiorczy PDF: {output_pdf_path}")

def main():
    output_dir = Path("results/plots_all_models")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel("data_loader/AnalysisWithOtolithPhoto_with_preds_KUMULACYJNE.xlsx")
    df = df[df["SET"].notna()]
    df = df[df["Wiek"] != -9]  # usunięcie przypadkow Wiek == -9
    df["Populacja"] = df["Populacja"].astype(int)
    df["Wiek"] = df["Wiek"].astype(int)

    for model_name, cols in model_to_column_map.items():
        pred_col = cols["pred"]
        prob_col = cols["prob"]
        if pred_col in df.columns:
            df[pred_col] = df[pred_col].astype("Int64")
            if prob_col and prob_col in df.columns:
                avg_prob = df[prob_col].mean()
                print(f"📊 Średnie probability dla {model_name}: {avg_prob:.4f}")
            show_table = model_name != "PRED_KUMULACYJNE"
            plot_correct_predictions_per_population(df, output_dir, model_name, pred_col, show_table)

    plot_accuracy_line_chart(df, output_dir)
    pdf_output_path = output_dir / "wszystkie_wykresy.pdf"
    save_all_plots_to_pdf(output_dir, pdf_output_path)

if __name__ == "__main__":
    main()
