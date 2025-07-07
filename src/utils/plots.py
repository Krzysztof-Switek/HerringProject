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


def generate_all_plots(metrics_df, log_dir):

    images = []

    tmp_acc = Path(log_dir) / "__temp_acc.png"
    tmp_loss = Path(log_dir) / "__temp_loss.png"

    plot_training_accuracy(metrics_df, tmp_acc)
    plot_training_loss(metrics_df, tmp_loss)

    images.append(tmp_acc)
    images.append(tmp_loss)

    # Dodawaj kolejne wykresy poniżej (np. F1, precision, multitask-specific etc.)
    # tmp_f1 = Path(log_dir) / "__temp_f1.png"
    # plot_training_f1(metrics_df, tmp_f1)
    # images.append(tmp_f1)

    return images
