import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

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


def generate_all_plots(metrics_df, cm_data, class_names, log_dir):
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

        # Nowe wykresy
        print(f"[DEBUG] Sprawdzam warunek dla 'Val Composite Score': {'Val Composite Score' in metrics_df.columns}")
        if 'Val Composite Score' in metrics_df.columns:
            tmp_comp_score = log_dir / "__temp_composite_score.png"
            plot_single_metric(metrics_df, tmp_comp_score, 'Val Composite Score', 'Composite Score (Walidacja)',
                               'Score')
            images.append(tmp_comp_score)

        print(f"[DEBUG] Sprawdzam warunek dla 'Val MAE Age': {'Val MAE Age' in metrics_df.columns}")
        if 'Val MAE Age' in metrics_df.columns:
            tmp_mae = log_dir / "__temp_mae_age.png"
            plot_single_metric(metrics_df, tmp_mae, 'Val MAE Age', 'MAE Age (Walidacja)', 'MAE')
            images.append(tmp_mae)

        loss_comp_cols = ["Train Classification Loss", "Val Classification Loss", "Train Regression Loss",
                          "Val Regression Loss"]
        print(
            f"[DEBUG] Sprawdzam warunek dla strat składowych: {all(col in metrics_df.columns for col in loss_comp_cols)}")
        if all(col in metrics_df.columns for col in loss_comp_cols):
            tmp_loss_comp = log_dir / "__temp_loss_components.png"
            plot_loss_components(metrics_df, tmp_loss_comp)
            images.append(tmp_loss_comp)

    print(f"[DEBUG] generate_all_plots zwraca {len(images)} obrazów.")
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
