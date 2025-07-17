import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from .report_constants import (
    USABLE_WIDTH,
    MATPLOTLIB_DEFAULTS,
    MATPLOT_FIG_HEIGHT,
)

# Ustaw parametry matplotlib na poziomie całego pliku
matplotlib.rcParams.update(MATPLOTLIB_DEFAULTS)


def plot_training_accuracy(metrics_df, save_path):
    """Wykres accuracy (train/val) na podstawie df z metrykami"""
    fig_width_inch = USABLE_WIDTH / 2 / 25.4  # połowa szerokości strony, w calach
    fig_height_inch = MATPLOT_FIG_HEIGHT
    dpi = MATPLOTLIB_DEFAULTS.get("figure.dpi", 150)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if metrics_df is not None and not metrics_df.empty:
        epochs = list(range(1, len(metrics_df) + 1))
        plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=dpi)
        plt.plot(epochs, metrics_df["Train Accuracy"], label="Train Acc", marker="o")
        plt.plot(epochs, metrics_df["Val Accuracy"], label="Val Acc", marker="o")
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Epoka")
        plt.title("Accuracy (trening/validacja)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    else:
        _plot_placeholder(save_path, "Brak danych metryk!", fig_width_inch, fig_height_inch)


def plot_training_loss(metrics_df, save_path):
    """Wykres loss (train/val) na podstawie df z metrykami"""
    fig_width_inch = USABLE_WIDTH / 2 / 25.4
    fig_height_inch = MATPLOT_FIG_HEIGHT
    dpi = MATPLOTLIB_DEFAULTS.get("figure.dpi", 150)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if metrics_df is not None and not metrics_df.empty:
        epochs = list(range(1, len(metrics_df) + 1))
        plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=dpi)
        plt.plot(epochs, metrics_df["Train Loss"], label="Train Loss", marker="o")
        plt.plot(epochs, metrics_df["Val Loss"], label="Val Loss", marker="o")
        plt.ylabel("Loss")
        plt.xlabel("Epoka")
        plt.title("Loss (trening/validacja)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    else:
        _plot_placeholder(save_path, "Brak danych metryk!", fig_width_inch, fig_height_inch)


def _plot_placeholder(save_path, message, width_inch, height_inch):
    """Tworzy wykres-zastępczy gdy brak danych."""
    plt.figure(figsize=(width_inch, height_inch))
    plt.text(0.5, 0.5, message, ha="center", va="center", fontsize=18, color="red")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_single_metric(metrics_df, save_path, column_name, title, y_label):
    """Generyczna funkcja do rysowania pojedynczej metryki w czasie."""
    fig_width_inch = USABLE_WIDTH / 2 / 25.4
    fig_height_inch = MATPLOT_FIG_HEIGHT
    dpi = MATPLOTLIB_DEFAULTS.get("figure.dpi", 150)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if metrics_df is not None and not metrics_df.empty and column_name in metrics_df.columns:
        epochs = list(range(1, len(metrics_df) + 1))
        plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=dpi)
        plt.plot(epochs, metrics_df[column_name], label=column_name, marker="o")
        plt.ylabel(y_label)
        plt.xlabel("Epoka")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    else:
        _plot_placeholder(save_path, f"Brak danych dla\\n{column_name}", fig_width_inch, fig_height_inch)


def plot_train_val_metric(metrics_df, save_path, metric_base_name, title, y_label):
    """Generyczna funkcja do rysowania metryki w czasie dla treningu i walidacji."""
    fig_width_inch = USABLE_WIDTH / 2 / 25.4
    fig_height_inch = MATPLOT_FIG_HEIGHT
    dpi = MATPLOTLIB_DEFAULTS.get("figure.dpi", 150)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    train_col = f"Train {metric_base_name}"
    val_col = f"Val {metric_base_name}"

    if metrics_df is not None and not metrics_df.empty and train_col in metrics_df.columns and val_col in metrics_df.columns:
        epochs = list(range(1, len(metrics_df) + 1))
        plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=dpi)
        plt.plot(epochs, metrics_df[train_col], label=f"Train {metric_base_name}", marker="o")
        plt.plot(epochs, metrics_df[val_col], label=f"Val {metric_base_name}", marker="o")
        plt.ylabel(y_label)
        plt.xlabel("Epoka")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    else:
        _plot_placeholder(save_path, f"Brak danych dla\\n{metric_base_name}", fig_width_inch, fig_height_inch)


def plot_loss_components(metrics_df, save_path):
    """Wykres strat składowych (train/val, classification/regression) na podstawie df z metrykami."""
    fig_width_inch = USABLE_WIDTH / 2 / 25.4
    fig_height_inch = MATPLOT_FIG_HEIGHT
    dpi = MATPLOTLIB_DEFAULTS.get("figure.dpi", 150)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    required_cols = [
        "Train Classification Loss", "Val Classification Loss",
        "Train Regression Loss", "Val Regression Loss"
    ]
    if metrics_df is not None and not metrics_df.empty and all(col in metrics_df.columns for col in required_cols):
        epochs = list(range(1, len(metrics_df) + 1))
        plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=dpi)

        plt.plot(epochs, metrics_df["Train Classification Loss"], label="Train Class. Loss", marker="o", linestyle='--')
        plt.plot(epochs, metrics_df["Val Classification Loss"], label="Val Class. Loss", marker="o")
        plt.plot(epochs, metrics_df["Train Regression Loss"], label="Train Reg. Loss", marker="x", linestyle='--')
        plt.plot(epochs, metrics_df["Val Regression Loss"], label="Val Reg. Loss", marker="x")

        plt.ylabel("Loss")
        plt.xlabel("Epoka")
        plt.title("Straty składowe (trening/walidacja)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    else:
        _plot_placeholder(save_path, "Brak danych\\nstrat składowych", fig_width_inch, fig_height_inch)


def plot_bayesian_optimization_weights(log_dir, save_path):
    """Wykres wag optymalizacji bayesowskiej w czasie."""
    fig_width_inch = USABLE_WIDTH / 2 / 25.4
    fig_height_inch = MATPLOT_FIG_HEIGHT
    dpi = MATPLOTLIB_DEFAULTS.get("figure.dpi", 150)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "bayesian_optimization_log.csv"
    if log_file.exists():
        try:
            df = pd.read_csv(log_file)
            plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=dpi)
            plt.plot(df["epoch"], df["alpha"], label="Alpha (F1 Global)", marker="o")
            plt.plot(df["epoch"], df["beta"], label="Beta (MAE Age)", marker="o")
            plt.plot(df["epoch"], df["gamma"], label="Gamma (F1 Subgroup)", marker="o")
            plt.ylabel("Waga")
            plt.xlabel("Epoka")
            plt.title("Wagi optymalizacji Bayesowskiej")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=dpi)
            plt.close()
        except Exception as e:
            print(f"Error plotting Bayesian optimization weights: {e}")
            _plot_placeholder(save_path, "Błąd w danych\\noptymalizacji Bayesowskiej", fig_width_inch, fig_height_inch)
    else:
        _plot_placeholder(save_path, "Brak danych\\noptymalizacji Bayesowskiej", fig_width_inch, fig_height_inch)


def generate_all_plots(metrics_df, cm_data, class_names, log_dir, predictions_df=None):
    """Generuje wszystkie wykresy dla raportu i zwraca listę ścieżek do plików."""
    print("[DEBUG] Uruchomiono generate_all_plots.")
    images = []
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Macierz pomyłek (jeśli dane są dostępne)
    print(f"[DEBUG] Sprawdzam macierz pomyłek: Dane dostępne? {cm_data is not None}")
    if cm_data is not None:
        tmp_cm = log_dir / "__temp_cm.png"
        plot_confusion_matrix(cm_data, class_names, tmp_cm)
        images.append(tmp_cm)

    # 2. Wykresy metryk z DataFrame
    if metrics_df is not None and not metrics_df.empty:
        # Istniejące wykresy
        tmp_acc = log_dir / "__temp_acc.png"
        plot_training_accuracy(metrics_df, tmp_acc)
        images.append(tmp_acc)

        tmp_loss = log_dir / "__temp_loss.png"
        plot_training_loss(metrics_df, tmp_loss)
        images.append(tmp_loss)

        # Nowe wykresy metryk (train/val)
        metrics_to_plot = {
            "F1": "F1 Score",
            "MAE Age": "MAE Age",
            "F1 Pop2 Age3-6": "F1 Subgroup",
            "Composite Score": "Composite Score"
        }

        for metric, title in metrics_to_plot.items():
            tmp_path = log_dir / f"__temp_{metric.lower().replace(' ', '_')}.png"
            plot_train_val_metric(metrics_df, tmp_path, metric, f"{title} (Train/Val)", title)
            images.append(tmp_path)

        loss_comp_cols = ["Train Classification Loss", "Val Classification Loss", "Train Regression Loss",
                          "Val Regression Loss"]
        print(
            f"[DEBUG] Sprawdzam warunek dla strat składowych: {all(col in metrics_df.columns for col in loss_comp_cols)}")
        if all(col in metrics_df.columns for col in loss_comp_cols):
            tmp_loss_comp = log_dir / "__temp_loss_components.png"
            plot_loss_components(metrics_df, tmp_loss_comp)
            images.append(tmp_loss_comp)

    # Wykres wag optymalizacji Bayesowskiej
    tmp_bayes_opt = log_dir / "__temp_bayes_opt.png"
    plot_bayesian_optimization_weights(log_dir, tmp_bayes_opt)
    images.append(tmp_bayes_opt)

    # Histogramy klasyfikacji
    images.extend(plot_classification_histograms(predictions_df, log_dir))

    print(f"[DEBUG] generate_all_plots zwraca {len(images)} obrazów.")
    return images


def plot_classification_histograms(predictions_df, log_dir):
    """Generuje histogramy poprawności klasyfikacji dla każdej populacji."""
    if predictions_df is None or predictions_df.empty:
        return []

    images = []
    populations = predictions_df["Population"].unique()
    for pop in populations:
        save_path = log_dir / f"__temp_hist_pop_{pop}.png"
        fig_width_inch = USABLE_WIDTH / 2 / 25.4
        fig_height_inch = MATPLOT_FIG_HEIGHT
        dpi = MATPLOTLIB_DEFAULTS.get("figure.dpi", 150)

        pop_df = predictions_df[predictions_df["Population"] == pop].copy()
        pop_df["Correct"] = pop_df["Population"] == pop_df["Prediction"]

        # Ensure age is numeric
        pop_df["Age"] = pd.to_numeric(pop_df["Age"], errors='coerce')
        pop_df.dropna(subset=["Age"], inplace=True)
        pop_df["Age"] = pop_df["Age"].astype(int)


        age_groups = sorted(pop_df["Age"].unique())
        correct_counts = pop_df[pop_df["Correct"]].groupby("Age").size()
        incorrect_counts = pop_df[~pop_df["Correct"]].groupby("Age").size()

        counts = pd.DataFrame({
            "Correct": correct_counts,
            "Incorrect": incorrect_counts
        }).reindex(age_groups).fillna(0)

        plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=dpi)
        counts.plot(kind="bar", stacked=True, color=["green", "red"])
        plt.title(f"Poprawność klasyfikacji dla Populacji {pop}")
        plt.xlabel("Grupa wiekowa")
        plt.ylabel("Liczba obserwacji")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.close()
        images.append(save_path)
    return images

def plot_confusion_matrix(cm_data, class_names, output_path, figsize=(8, 6)):
    """
    Generuje i zapisuje wykres macierzy pomyłek.
    """
    import seaborn as sns
    import numpy as np

    plt.figure(figsize=figsize)

    # Sprawdzenie, czy suma wiersza nie jest zerem, aby uniknąć dzielenia przez zero
    cm_sum = cm_data.sum(axis=1)[:, np.newaxis]
    # Użyj np.divide z klauzulą 'where', aby uniknąć warningów dla wierszy z samymi zerami
    cm_normalized = np.divide(cm_data.astype('float'), cm_sum, out=np.zeros_like(cm_data, dtype=float),
                              where=(cm_sum != 0))

    # Formatowanie adnotacji, aby pokazać zarówno liczbę, jak i procent
    annot = np.empty_like(cm_data).astype(str)
    rows, cols = cm_data.shape
    for r in range(rows):
        for c in range(cols):
            count = cm_data[r, c]
            percent = cm_normalized[r, c]
            if count == 0 and percent == 0:
                annot[r, c] = '0'
            else:
                annot[r, c] = f'{count}\\n({percent:.1%})'

    sns.heatmap(
        cm_normalized,
        annot=annot,
        fmt='',  # Używamy pustego formatu, bo adnotacje są już sformatowanymi stringami
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        vmin=0.0,  # Ustawienie zakresu paska kolorów
        vmax=1.0
    )
    plt.title('Macierz pomyłek (dla najlepszej epoki wg Composite Score)')
    plt.ylabel('Prawdziwa etykieta')
    plt.xlabel('Przewidziana etykieta')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path)
