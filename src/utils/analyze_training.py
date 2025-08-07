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


# ===================================================================
# 🔧 USTAWIENIA UŻYTKOWNIKA
# Podaj domyślną ścieżkę do katalogu z logami, który chcesz analizować.
# Ta ścieżka zostanie użyta, jeśli skrypt zostanie uruchomiony bez
# podawania argumentu w wierszu poleceń.
DEFAULT_LOG_DIR = "C:/Users/kswitek/Documents/HerringProject/results/logs/BEST_resnet50_standard_ce_multi_2025-07-19_19-24"
# ===================================================================


def parse_arguments():
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Analizuje i wizualizuje wyniki treningu z podanego katalogu logów.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Przykłady użycia:
  1. Użycie domyślnej ścieżki z pliku:
     python src/utils/analyze_training.py

  2. Podanie konkretnej ścieżki:
     python src/utils/analyze_training.py results/logs/resnet50_standard_ce_multi_2023-05-30_10-00
"""
    )
    parser.add_argument(
        "log_dir",
        type=str,
        nargs='?',  # Argument jest opcjonalny
        default=None,  # Domyślna wartość, jeśli nie podano
        help="Opcjonalna ścieżka do katalogu z logami. Jeśli nie podano, użyta zostanie domyślna z pliku."
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
    metrics_df = None
    # --- Wczytywanie metryk treningowych ---
    metrics_files = glob.glob(str(log_dir / "*training_metrics.csv"))
    if not metrics_files:
        print(f"  - ⚠️ Nie znaleziono pliku metryk (*training_metrics.csv). Wykresy metryk nie będą dostępne.")
    else:
        try:
            metrics_path = Path(metrics_files[0])
            print(f"  - Znaleziono plik metryk: {metrics_path.name}")
            metrics_df = pd.read_csv(metrics_path)
        except Exception as e:
            print(f"  - ❌ Błąd podczas wczytywania pliku metryk: {e}")
            metrics_df = None


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
    if metrics_df is None:
        return None
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


def plot_composite_score(metrics_df: pd.DataFrame):
    """
    Generuje dedykowany wykres dla Val Composite Score.
    """
    if metrics_df is None or 'Val Composite Score' not in metrics_df.columns or metrics_df['Val Composite Score'].isna().all():
        return None

    df = metrics_df.copy()
    df['epoch_num'] = df['Epoch'].str.extract(r'e(\d+)').astype(int)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(df['epoch_num'], df['Val Composite Score'], 'o-', label='Validation', color='purple')
    ax.set_title('Validation Composite Score Over Epochs')
    ax.set_ylabel('Composite Score')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # Znajdź i oznacz najlepszy wynik
    best_epoch_idx = df['Val Composite Score'].idxmax()
    best_score = df['Val Composite Score'].max()
    best_epoch_num = df['epoch_num'][best_epoch_idx]
    ax.annotate(f'Best: {best_score:.4f}',
                xy=(best_epoch_num, best_score),
                xytext=(best_epoch_num, best_score * 0.9),
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='center')


    fig.suptitle("Dedykowany wykres: Val Composite Score", fontsize=16)
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
    run_name = log_dir.name

    summary_text = f"""
    ANALIZOWANY PRZEBIEG
    -----------------------------------
    Katalog logów: {run_name}
    """

    if metrics_df is None:
        summary_text += """


    BŁĄD KRYTYCZNY
    -----------------------------------
    Nie znaleziono pliku z metrykami treningowymi (*_training_metrics.csv).
    Większość analizy jest niemożliwa do przeprowadzenia.

    Sprawdź, czy w podanym katalogu znajduje się poprawny plik z metrykami.
    """
        ax.text(0.05, 0.95, summary_text, fontsize=11, va='top', family='monospace', color='red')
        return fig


    best_epoch_data = None
    best_metric_name = "N/A"
    best_epoch_num = "N/A"

    # --- Logika wyboru najlepszej epoki ---
    if 'Val Composite Score' in metrics_df.columns and metrics_df['Val Composite Score'].notna().any():
        try:
            best_epoch_idx = metrics_df['Val Composite Score'].idxmax()
            best_epoch_data = metrics_df.loc[best_epoch_idx]
            best_metric_name = 'Val Composite Score'
        except ValueError:
            best_epoch_data = None
    if best_epoch_data is None and 'Val F1' in metrics_df.columns and metrics_df['Val F1'].notna().any():
        try:
            best_epoch_idx = metrics_df['Val F1'].idxmax()
            best_epoch_data = metrics_df.loc[best_epoch_idx]
            best_metric_name = 'Val F1'
        except ValueError:
            best_epoch_data = None
    if best_epoch_data is None and not metrics_df.empty:
        best_epoch_data = metrics_df.iloc[-1]
        best_metric_name = "Last Epoch (no primary metric available)"

    total_epochs = metrics_df['Epoch'].nunique()
    summary_text += f"""
    Całkowita liczba epok: {total_epochs}
    """

    if best_epoch_data is not None:
        epoch_str_parts = best_epoch_data['Epoch'].split('-e')
        if len(epoch_str_parts) > 1:
            best_epoch_num = epoch_str_parts[-1]
        else:
            best_epoch_num = "N/A"

        summary_text += f"""


    NAJLEPSZY WYNIK (wg: {best_metric_name})
    -----------------------------------
    Osiągnięty w epoce: {best_epoch_num}
"""
        key_metrics = {
            'Val Composite Score': 'Val Composite Score',
            'Val F1': 'Val F1 (globalny)',
            'Val MAE Age': 'Val MAE Age',
            'Val F1 Pop2 Age3-6': 'Val F1 Pop2 Age3-6'
        }
        summary_text += "\n\n    Kluczowe metryki w najlepszej epoce:\n"
        for col, desc in key_metrics.items():
            if col in best_epoch_data and pd.notna(best_epoch_data[col]):
                summary_text += f"      - {desc:<20}: {best_epoch_data[col]:.4f}\n"

        other_metrics = {
            'Val Accuracy': 'Val Accuracy',
            'Val Loss': 'Val Loss',
            'Val AUC': 'Val AUC'
        }
        summary_text += "\n    Pozostałe metryki w najlepszej epoce:\n"
        for col, desc in other_metrics.items():
            if col in best_epoch_data and pd.notna(best_epoch_data[col]):
                value = best_epoch_data[col]
                format_str = "{:.2f}%" if 'Accuracy' in col else "{:.4f}"
                summary_text += f"      - {desc:<20}: {format_str.format(value)}\n"
    else:
        summary_text += "\n\n    Nie można było wyznaczyć najlepszej epoki (brak danych)."

    ax.text(0.05, 0.95, summary_text, fontsize=10, va='top', family='monospace')
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
    # Usunięto hardkodowane kolory, aby umożliwić obsługę więcej niż 2 populacji
    pop_counts.plot(kind='bar', ax=ax)
    ax.set_title('Całkowita liczba augmentacji per populacja')
    ax.set_xlabel('Populacja')
    ax.set_ylabel('Liczba zastosowanych augmentacji')
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    return fig


def main():
    """Główna funkcja skryptu."""
    args = parse_arguments()

    if args.log_dir:
        log_dir = Path(args.log_dir)
        print(f"ℹ️ Używam ścieżki z wiersza poleceń: {log_dir}")
    else:
        log_dir = Path(DEFAULT_LOG_DIR)
        print(f"ℹ️ Używam domyślnej ścieżki z pliku: {log_dir}")

    if not log_dir.is_dir():
        print(f"Błąd: Podana ścieżka '{log_dir}' nie jest prawidłowym katalogiem.")
        print("Upewnij się, że ścieżka jest poprawna w sekcji USTAWIENIA UŻYTKOWNIKA lub podana jako argument.")
        return

    try:
        metrics_df, confusion_matrix, cm_labels, augment_df = load_data(log_dir)

        if metrics_df is not None:
             print("\n✅ Pomyślnie wczytano dane metryk. Ostatnie 3 wiersze:")
             print(metrics_df.tail(3).to_string())

        # Zawsze generuj stronę podsumowującą (nawet z błędem)
        summary_fig = create_summary_page(metrics_df, log_dir)

        # Generuj wizualizacje tylko jeśli dane są dostępne
        composite_score_fig = plot_composite_score(metrics_df)
        metrics_fig = plot_metrics(metrics_df)
        cm_fig = plot_confusion_matrix(confusion_matrix, cm_labels)
        augment_fig = plot_augmentation_summary(augment_df)

        pdf_path = log_dir / "training_analysis_report.pdf"
        print(f"\n🚀 Generowanie raportu PDF: {pdf_path}")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(summary_fig)
            if composite_score_fig:
                pdf.savefig(composite_score_fig)
            if metrics_fig:
                pdf.savefig(metrics_fig)
            if cm_fig:
                pdf.savefig(cm_fig)
            if augment_fig:
                pdf.savefig(augment_fig)

        print(f"📄 Raport został pomyślnie zapisany.")
        plt.show()

    except Exception as e:
        print(f"\nBłąd krytyczny: Wystąpił nieoczekiwany błąd: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
##