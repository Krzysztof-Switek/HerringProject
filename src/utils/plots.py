import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from .report_constants import USABLE_WIDTH, MATPLOTLIB_DEFAULTS, MATPLOT_FIG_HEIGHT

matplotlib.rcParams.update(MATPLOTLIB_DEFAULTS)

def generate_report_plots(metrics, log_dir):
    """
    Tworzy standardowe wykresy dla raportu i zwraca listę ścieżek do plików.
    Każdy wykres jest zapisywany w katalogu log_dir.
    """
    plot_files = []

    # ACCURACY
    acc_path = Path(log_dir) / "__temp_acc.png"
    fig_width_inch = USABLE_WIDTH / 2 / 25.4
    fig_height_inch = MATPLOT_FIG_HEIGHT
    dpi = MATPLOTLIB_DEFAULTS.get('figure.dpi', 150)

    if metrics is not None and not metrics.empty:
        epochs = list(range(1, len(metrics) + 1))
        # Accuracy plot
        plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=dpi)
        plt.plot(epochs, metrics["Train Accuracy"], label="Train Acc", marker='o')
        plt.plot(epochs, metrics["Val Accuracy"], label="Val Acc", marker='o')
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Epoka")
        plt.title("Accuracy (trening/validacja)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(acc_path, dpi=dpi)
        plt.close()
        plot_files.append(acc_path)

        # Loss plot
        loss_path = Path(log_dir) / "__temp_loss.png"
        plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=dpi)
        plt.plot(epochs, metrics["Train Loss"], label="Train Loss", marker='o')
        plt.plot(epochs, metrics["Val Loss"], label="Val Loss", marker='o')
        plt.ylabel("Loss")
        plt.xlabel("Epoka")
        plt.title("Loss (trening/validacja)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(loss_path, dpi=dpi)
        plt.close()
        plot_files.append(loss_path)

    else:
        # Placeholder for missing data
        from .plots import plot_placeholder  # Dla czytelności – zamieniasz na własną funkcję jeśli chcesz!
        acc_path = Path(log_dir) / "__temp_acc.png"
        loss_path = Path(log_dir) / "__temp_loss.png"
        plot_placeholder(acc_path, "Brak danych metryk!", fig_width_inch, fig_height_inch)
        plot_placeholder(loss_path, "Brak danych metryk!", fig_width_inch, fig_height_inch)
        plot_files.extend([acc_path, loss_path])

    return plot_files

def plot_placeholder(save_path, message, width_inch, height_inch):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(width_inch, height_inch))
    plt.text(0.5, 0.5, message, ha='center', va='center', fontsize=18, color='red')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
