"""
Skrypt do analizy wyników predykcji modelu.
"""

import pandas as pd
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# ===================================================================
# 🔧 USTAWIENIA UŻYTKOWNIKA
# ===================================================================

# 1. Podaj ścieżkę do katalogu z logami, który zawiera plik .xlsx z predykcjami.
LOG_DIR = "C:/Users/kswitek/Documents/HerringProject/results/logs/BEST_resnet50_standard_ce_multi_2025-07-19_19-24"

# 2. Zdefiniuj mapowanie nazw kolumn na ich adresy w pliku Excel.
#    Nazwy po lewej stronie są używane w skrypcie. Adresy po prawej to kolumny w Excelu.
COLUMN_MAPPING = {
    # Dane biologiczne
    "Wiek": "U",
    "Populacja": "AH",
    # Dane z modelu
    "SET": "AQ",
    "populacja_pred": "AR",
    "prediction_probability": "AS",
    "age_pred": "AT",
}


# ===================================================================

def excel_col_to_int(col_str: str) -> int:
    """Konwertuje nazwę kolumny Excela (np. 'A', 'AR') na indeks liczbowy (zaczynając od 0)."""
    index = 0
    for char in col_str:
        index = index * 26 + (ord(char.upper()) - ord('A')) + 1
    return index - 1


def create_prediction_summary_page(df: pd.DataFrame, log_dir: Path):
    """Tworzy stronę tytułową/podsumowującą dla raportu predykcji."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    run_name = log_dir.name
    num_rows = len(df)

    summary_text = f"""
    Raport z Analizy Predykcji
    ===================================

    Analizowany przebieg:
    --------------------
    Katalog logów: {run_name}


    Podstawowe statystyki:
    --------------------
    Liczba przeanalizowanych wierszy: {num_rows}
    (Po odfiltrowaniu wierszy bez 'SET' i z `Wiek == -9`)

    """

    ax.text(0.05, 0.95, summary_text, fontsize=12, va='top', family='monospace')
    return fig


def plot_population_confusion_matrices(df: pd.DataFrame):
    """
    Generuje 4 macierze pomyłek dla predykcji populacji na jednej stronie:
    Ogólną, dla zbioru TRAIN, VAL i TEST.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    datasets = ['Overall', 'TRAIN', 'VAL', 'TEST']

    fig, axes = plt.subplots(2, 2, figsize=(15, 15), constrained_layout=True)
    axes = axes.flatten()

    all_true_pop = df['Populacja'].astype(str)
    all_pred_pop = df['populacja_pred'].astype(str)
    all_labels = sorted(list(set(all_true_pop) | set(all_pred_pop)))

    for i, dataset_name in enumerate(datasets):
        ax = axes[i]

        if dataset_name == 'Overall':
            subset_df = df
            title = 'Macierz pomyłek (Ogólna)'
        else:
            subset_df = df[df['SET'] == dataset_name]
            title = f'Macierz pomyłek (SET: {dataset_name})'

        if subset_df.empty:
            ax.text(0.5, 0.5, 'Brak danych', ha='center', va='center')
            ax.set_title(title)
            ax.axis('off')
            continue

        true_pop = subset_df['Populacja'].astype(str)
        pred_pop = subset_df['populacja_pred'].astype(str)

        cm = confusion_matrix(true_pop, pred_pop, labels=all_labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
        disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')

        ax.set_title(title)
        ax.set_xlabel('Populacja przewidziana (populacja_pred)')
        ax.set_ylabel('Populacja prawdziwa (Populacja)')

    fig.suptitle('Macierze pomyłek dla predykcji populacji', fontsize=16)
    return fig


def plot_age_scatter(df: pd.DataFrame):
    """
    Generuje wykresy rozrzutu wieku rzeczywistego vs. przewidywanego dla każdej populacji.
    """
    populations = df['Populacja'].unique()
    num_plots = len(populations)

    # Tworzenie siatki wykresów
    # Prosta heurystyka: jeśli mamy 2 lub 3 populacje, pokaż w jednym wierszu. Jeśli więcej, zrób 2 kolumny.
    if num_plots <= 3:
        num_rows, num_cols = 1, num_plots
        figsize = (6 * num_cols, 5)
    else:
        num_cols = 2
        num_rows = (num_plots + 1) // 2
        figsize = (12, 5 * num_rows)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, constrained_layout=True, squeeze=False)
    axes = axes.flatten()

    for i, pop in enumerate(populations):
        ax = axes[i]
        pop_df = df[df['Populacja'] == pop]

        # Użycie rasterized=True, aby zmniejszyć rozmiar pliku PDF
        ax.scatter(pop_df['Wiek'], pop_df['age_pred'], alpha=0.6, rasterized=True)

        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Idealna predykcja')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_title(f'Populacja {pop}')
        ax.set_xlabel('Wiek biologiczny (Wiek)')
        ax.set_ylabel('Wiek przewidziany (age_pred)')
        ax.legend()
        ax.grid(True)

    # Ukryj puste osie, jeśli istnieją
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Wiek biologiczny vs. Wiek przewidziany przez model', fontsize=16)
    return fig


def plot_probability_by_age_class(df: pd.DataFrame):
    """
    Generuje wykresy pudełkowe rozkładu prawdopodobieństw dla każdej populacji i klasy wiekowej.
    """
    populations = sorted(df['Populacja'].unique())
    num_plots = len(populations)

    if num_plots == 0:
        return None

    # Dynamiczne tworzenie siatki wykresów
    num_cols = 2 if num_plots > 1 else 1
    num_rows = (num_plots + num_cols - 1) // num_cols
    figsize = (12, 6 * num_rows)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, constrained_layout=True, squeeze=False)
    axes = axes.flatten()

    for i, pop in enumerate(populations):
        ax = axes[i]
        pop_df = df[df['Populacja'] == pop].copy()

        # Sortowanie wg wieku, aby osie X były uporządkowane
        sorted_ages = sorted(pop_df['Wiek'].unique())

        sns.boxplot(x='Wiek', y='prediction_probability', data=pop_df, ax=ax, order=sorted_ages)

        ax.set_title(f'Rozkład prawdopodobieństwa dla Populacji {pop}')
        ax.set_xlabel('Klasa wiekowa (Wiek)')
        ax.set_ylabel('Prawdopodobieństwo predykcji (prediction_probability)')
        ax.tick_params(axis='x', rotation=45)

    # Ukryj puste osie
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Rozkład prawdopodobieństwa predykcji wg populacji i wieku', fontsize=16)
    return fig


def plot_age_confusion_matrices_per_population(df: pd.DataFrame):
    """Generuje macierze pomyłek dla predykcji wieku dla każdej populacji na jednej stronie."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    populations = sorted(df['Populacja'].unique())

    if not populations:
        return None

    num_plots = len(populations)
    fig, axes = plt.subplots(1, num_plots, figsize=(10 * num_plots, 10), constrained_layout=True)
    if num_plots == 1:
        axes = [axes]  # make it iterable

    for i, pop in enumerate(populations):
        ax = axes[i]
        pop_df = df[df['Populacja'] == pop]
        true_age = pop_df['Wiek'].astype(int)
        pred_age = pop_df['age_pred'].astype(int)

        labels = sorted(list(set(true_age) | set(pred_age)))

        cm = confusion_matrix(true_age, pred_age, labels=labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap='Greens', text_kw={'color': 'black'})

        # Podświetlenie przekątnej
        for r in range(len(labels)):
            for c in range(len(labels)):
                if r == c:
                    disp.text_[r, c].set_weight('bold')
                    disp.text_[r, c].set_color('blue')

        ax.set_title(f'Macierz pomyłek wieku - Populacja {pop}')
        ax.set_xlabel('Wiek przewidziany (age_pred)')
        ax.set_ylabel('Wiek prawdziwy (Wiek)')

    fig.suptitle('Macierze pomyłek dla predykcji wieku wg populacji', fontsize=16)
    return fig


def plot_correctness_by_age_stacked_bar(df: pd.DataFrame):
    """
    Generuje skumulowane wykresy słupkowe pokazujące poprawność predykcji populacji
    w zależności od wieku, dla każdej populacji osobno, na jednej stronie.
    """
    populations = sorted(df['Populacja'].unique())

    if not populations:
        return None

    num_plots = len(populations)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 7 * num_plots), constrained_layout=True)
    if num_plots == 1:
        axes = [axes]  # make it iterable

    for i, pop in enumerate(populations):
        ax = axes[i]
        pop_df = df[df['Populacja'] == pop].copy()
        pop_df['is_correct'] = pop_df['Populacja'] == pop_df['populacja_pred']

        counts = pop_df.groupby(['Wiek', 'is_correct']).size().unstack(fill_value=0)
        counts.rename(columns={True: 'Poprawne', False: 'Błędne'}, inplace=True)

        if 'Błędne' not in counts:
            counts['Błędne'] = 0
        if 'Poprawne' not in counts:
            counts['Poprawne'] = 0

        counts.plot(kind='bar', stacked=True, color={'Poprawne': 'green', 'Błędne': 'red'}, ax=ax)

        ax.set_title(f'Poprawność predykcji populacji wg wieku - Populacja {pop}')
        ax.set_xlabel('Wiek biologiczny (Wiek)')
        ax.set_ylabel('Liczba przypadków')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Typ predykcji')

    fig.suptitle('Poprawność predykcji populacji wg wieku', fontsize=16)
    return fig


def main():
    """Główna funkcja skryptu."""
    log_dir = Path(LOG_DIR)

    if not log_dir.is_dir():
        print(f"Błąd: Podana ścieżka '{log_dir}' nie jest prawidłowym katalogiem.")
        return

    xlsx_files = glob.glob(str(log_dir / "*.xlsx"))
    if not xlsx_files:
        print(f"Błąd: Nie znaleziono pliku .xlsx w katalogu '{log_dir}'.")
        return

    predictions_path = Path(xlsx_files[0])
    print(f"Znaleziono plik z predykcjami: {predictions_path.name}")

    try:
        col_indices = [excel_col_to_int(v) for v in COLUMN_MAPPING.values()]
        col_names = list(COLUMN_MAPPING.keys())

        df = pd.read_excel(
            predictions_path,
            engine='openpyxl',
            header=None,
            usecols=col_indices
        )

        map_to_new_names = {df.columns[i]: col_names[i] for i in range(len(col_names))}
        df.rename(columns=map_to_new_names, inplace=True)

        # Usuń pierwszy wiersz, który jest starym nagłówkiem
        df = df.drop(df.index[0]).reset_index(drop=True)

        # Konwertuj kolumny na typy numeryczne
        numeric_cols = ['Wiek', 'Populacja', 'populacja_pred', 'prediction_probability', 'age_pred']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Usuń wiersze, gdzie konwersja na liczbę się nie powiodła (jeśli jakieś są)
        df.dropna(subset=numeric_cols, inplace=True)

        # Filtruj dane, gdzie Wiek != -9
        df = df[df['Wiek'] != -9].copy()

        df_filtered = df[df['SET'].notna()].copy()

        if df_filtered.empty:
            print("Błąd: Brak danych do analizy po odfiltrowaniu.")
            return

        print("\nPrzetworzone dane (pierwsze 5 wierszy):")
        print(df_filtered.head().to_string())

        # Generowanie wizualizacji
        print("\nGenerowanie wizualizacji...")
        summary_fig = create_prediction_summary_page(df_filtered, log_dir)
        cm_fig = plot_population_confusion_matrices(df_filtered)
        age_scatter_fig = plot_age_scatter(df_filtered)
        prob_by_age_fig = plot_probability_by_age_class(df_filtered)
        age_cm_fig = plot_age_confusion_matrices_per_population(df_filtered)
        correctness_fig = plot_correctness_by_age_stacked_bar(df_filtered)
        print("Wizualizacje wygenerowane.")

        # Zapis do pliku PDF
        pdf_path = log_dir / "prediction_analysis_report.pdf"
        print(f"\n🚀 Generowanie raportu PDF: {pdf_path}")
        with PdfPages(pdf_path) as pdf:
            # Zapis z DPI=150 w celu optymalizacji rozmiaru pliku
            if summary_fig:
                pdf.savefig(summary_fig)
            if cm_fig:
                pdf.savefig(cm_fig, dpi=150)
            if age_scatter_fig:
                pdf.savefig(age_scatter_fig, dpi=150)
            if prob_by_age_fig:
                pdf.savefig(prob_by_age_fig, dpi=150)
            if age_cm_fig:
                pdf.savefig(age_cm_fig, dpi=150)
            if correctness_fig:
                pdf.savefig(correctness_fig, dpi=150)

        print(f"📄 Raport został pomyślnie zapisany.")
        plt.show()


    except Exception as e:
        print(f"Wystąpił błąd podczas przetwarzania pliku: {e}")


if __name__ == "__main__":
    main()
