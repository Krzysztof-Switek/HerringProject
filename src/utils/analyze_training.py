"""
Skrypt do analizy i wizualizacji wyników treningu modelu.

Ten skrypt wczytuje logi z określonego folderu, generuje wykresy metryk treningowych
i walidacyjnych, wizualizuje macierz pomyłek oraz podsumowanie augmentacji,
a następnie zapisuje wszystko w postaci raportu PDF.

Użycie:
    python src/utils/analyze_training.py /path/to/your/log_directory

Przykład:
    python src/utils/analyze_training.py results/logs/resnet50_standard_ce_multi_2023-05-30_10-00
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import glob


def parse_arguments():
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Analizuje i wizualizuje wyniki treningu z podanego katalogu logów.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Przykład użycia:
  python src/utils/analyze_training.py results/logs/resnet50_standard_ce_multi_2023-05-30_10-00
"""
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Ścieżka do katalogu z logami treningowymi, który ma być przeanalizowany."
    )
    return parser.parse_args()

def load_data(log_dir: Path):
    """
    Wczytuje wszystkie niezbędne pliki z katalogu logów.

    Args:
        log_dir: Ścieżka do katalogu z logami.

    Returns:
        Tuple: (metrics_df, confusion_matrix, cm_labels, augment_df)
    """
    print(f"🔍 Przeszukiwanie katalogu: {log_dir}")

    # --- Wczytywanie metryk treningowych ---
    metrics_files = glob.glob(str(log_dir / "*_training_metrics.csv"))
    if not metrics_files:
        raise FileNotFoundError(f"Nie znaleziono pliku metryk (*_training_metrics.csv) w {log_dir}")
    metrics_path = Path(metrics_files[0])
    print(f"  - Znaleziono plik metryk: {metrics_path.name}")
    metrics_df = pd.read_csv(metrics_path)

    # --- Wczytywanie macierzy pomyłek ---
    cm_path = log_dir / "best_confusion_matrix.csv"
    confusion_matrix, cm_labels = None, None
    if cm_path.exists():
        print(f"  - Znaleziono macierz pomyłek: {cm_path.name}")
        cm_df = pd.read_csv(cm_path, header=None)
        confusion_matrix = cm_df.values
        # Zakładamy, że etykiety to numery populacji, można je dostosować w razie potrzeby
        cm_labels = [str(i) for i in range(confusion_matrix.shape[0])]
    else:
        print("  - ⚠️ Nie znaleziono pliku best_confusion_matrix.csv.")

    # --- Wczytywanie podsumowania augmentacji ---
    augment_files = glob.glob(str(log_dir / "augmentation_summary_*.csv"))
    augment_df = None
    if augment_files:
        augment_path = Path(augment_files[0])
        print(f"  - Znaleziono podsumowanie augmentacji: {augment_path.name}")
        augment_df = pd.read_csv(augment_path)
    else:
        print("  - ⚠️ Nie znaleziono pliku podsumowania augmentacji (augmentation_summary_*.csv).")

    return metrics_df, confusion_matrix, cm_labels, augment_df

def plot_metrics(metrics_df: pd.DataFrame):
    """
    Generuje wykresy metryk treningowych i walidacyjnych.
    """
    # --- Przygotowanie danych ---
    df = metrics_df.copy()
    # Wyodrębnienie numeru epoki z formatu 'nazwa-eX'
    df['epoch_num'] = df['Epoch'].str.extract(r'e(\d+)').astype(int)

    # Definicja metryk do wyplotowania
    plots_definitions = {
        'Accuracy': ('Train Accuracy', 'Val Accuracy'),
        'F1 Score': ('Train F1', 'Val F1'),
        'Loss': ('Train Loss', 'Val Loss'),
        'AUC': ('Train AUC', 'Val AUC'),
        'Composite Score': (None, 'Val Composite Score'),
        'MAE Age': (None, 'Val MAE Age'),
        'F1 Pop2 Age3-6': (None, 'Val F1 Pop2 Age3-6'),
        'Classification Loss': ('Train Classification Loss', 'Val Classification Loss'),
        'Regression Loss': ('Train Regression Loss', 'Val Regression Loss'),
    }

    # Ustalenie rozmiaru siatki wykresów
    num_plots = len(plots_definitions)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8.5 * 2, 11 * 1.5), constrained_layout=True)
    axes = axes.flatten()

    plot_idx = 0
    for title, (train_col, val_col) in plots_definitions.items():
        ax = axes[plot_idx]
        if train_col and train_col in df.columns:
            ax.plot(df['epoch_num'], df[train_col], 'o-', label='Train')
        if val_col and val_col in df.columns:
            ax.plot(df['epoch_num'], df[val_col], 'o-', label='Validation')

        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)
        plot_idx += 1

    # Ukrycie pustych osi
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Wykresy metryk treningowych", fontsize=16, y=1.02)
    return fig

def plot_confusion_matrix(cm, labels):
    if cm is None:
        return None

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def create_summary_page(metrics_df: pd.DataFrame, log_dir: Path):
    """
    Tworzy stronę podsumowującą, zawierającą kluczowe informacje o najlepszym wyniku.
    """
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.set_title("Training Summary", fontsize=14)

    # Znajdź epokę z najlepszym `Val Composite Score`
    best_epoch_idx = metrics_df['Val Composite Score'].idxmax()
    best_epoch_data = metrics_df.loc[best_epoch_idx]

    run_name = log_dir.name
    total_epochs = metrics_df['Epoch'].nunique()
    best_epoch_num = int(best_epoch_data['Epoch'].split('-e')[-1])

    summary_text = f"""
    ANALIZOWANY PRZEBIEG
    -----------------------------------
    Katalog logów: {run_name}
    Całkowita liczba epok: {total_epochs}


    NAJLEPSZY WYNIK (wg Val Composite Score)
    -----------------------------------
    Osiągnięty w epoce: {best_epoch_num}

    Kluczowa metryka:
      Val Composite Score: {best_epoch_data['Val Composite Score']:.4f}

    Metryki składowe w najlepszej epoce:
      - Val F1 (globalny):   {best_epoch_data['Val F1']:.4f}
      - Val MAE Age:         {best_epoch_data['Val MAE Age']:.4f}
      - Val F1 Pop2 Age3-6:  {best_epoch_data['Val F1 Pop2 Age3-6']:.4f}

    Pozostałe metryki w najlepszej epoce:
      - Val Accuracy:        {best_epoch_data['Val Accuracy']:.2f}%
      - Val Loss:            {best_epoch_data['Val Loss']:.4f}
      - Val AUC:             {best_epoch_data['Val AUC']:.4f}
    """

    ax.text(0.05, 0.95, summary_text, fontsize=11, va='top', family='monospace')
    return fig


def plot_augmentation_summary(augment_df: pd.DataFrame):
    """
    Generuje wykresy podsumowujące użycie augmentacji.
    """
    if augment_df is None:
        return None

    # Zmiana: Użycie poprawnych nazw kolumn
    pop_counts = augment_df.groupby('Populacja')['Liczba_augmentacji'].sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    pop_counts.plot(kind='bar', ax=ax, color=['skyblue', 'orange'])
    ax.set_title('Całkowita liczba augmentacji per populacja')
    ax.set_xlabel('Populacja')
    ax.set_ylabel('Liczba zastosowanych augmentacji')
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    return fig


def main():
    """Główna funkcja skryptu."""
    args = parse_arguments()
    log_dir = Path(args.log_dir)

    if not log_dir.is_dir():
        print(f"Błąd: Podana ścieżka '{log_dir}' nie jest prawidłowym katalogiem.")
        return

    try:
        metrics_df, confusion_matrix, cm_labels, augment_df = load_data(log_dir)
        print("\n✅ Pomyślnie wczytano dane. Ostatnie 3 wiersze metryk:")
        print(metrics_df.tail(3).to_string())

        # Generowanie wizualizacji
        summary_fig = create_summary_page(metrics_df, log_dir)
        metrics_fig = plot_metrics(metrics_df)
        cm_fig = plot_confusion_matrix(confusion_matrix, cm_labels)
        augment_fig = plot_augmentation_summary(augment_df)

        # Zapis do pliku PDF
        pdf_path = log_dir / "training_analysis_report.pdf"
        print(f"\n🚀 Generowanie raportu PDF: {pdf_path}")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(summary_fig)
            if metrics_fig:
                pdf.savefig(metrics_fig)
            if cm_fig:
                pdf.savefig(cm_fig)
            if augment_fig:
                pdf.savefig(augment_fig)

        print(f"📄 Raport został pomyślnie zapisany.")

        # Opcjonalne wyświetlenie wykresów na ekranie
        plt.show()

    except FileNotFoundError as e:
        print(f"\nBłąd krytyczny: {e}")
    except Exception as e:
        print(f"\nWystąpił nieoczekiwany błąd: {e}")


if __name__ == "__main__":
    main()
##