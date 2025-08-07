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


def plot_population_confusion_matrix(df: pd.DataFrame):
    """Generuje macierz pomyłek dla predykcji populacji."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Upewnij się, że obie kolumny są tego samego typu, aby uniknąć błędów
    true_pop = df['Populacja'].astype(str)
    pred_pop = df['populacja_pred'].astype(str)

    # Znajdź wszystkie unikalne etykiety w danych, aby zapewnić spójność
    labels = sorted(list(set(true_pop) | set(pred_pop)))

    cm = confusion_matrix(true_pop, pred_pop, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')

    ax.set_title('Macierz pomyłek dla predykcji populacji')
    plt.tight_layout()
    return fig


def plot_probability_distribution(df: pd.DataFrame):
    """
    Generuje wykresy rozkładu prawdopodobieństw dla poprawnych i błędnych predykcji.
    """
    df['is_correct'] = df['Populacja'] == df['populacja_pred']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df, x='prediction_probability', hue='is_correct', fill=True, ax=ax)

    ax.set_title('Rozkład prawdopodobieństwa predykcji')
    ax.set_xlabel('Prawdopodobieństwo')
    ax.set_ylabel('Gęstość')
    plt.tight_layout()
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

        ax.scatter(pop_df['Wiek'], pop_df['age_pred'], alpha=0.6)

        # Linia y=x dla idealnej predykcji
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Idealna predykcja')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_title(f'Populacja {pop}')
        ax.set_xlabel('Wiek biologiczny')
        ax.set_ylabel('Wiek przewidziany')
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
        ax.set_xlabel('Klasa wiekowa')
        ax.set_ylabel('Prawdopodobieństwo predykcji')
        ax.tick_params(axis='x', rotation=45)

    # Ukryj puste osie
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Rozkład prawdopodobieństwa predykcji wg populacji i wieku', fontsize=16)
    return fig


def plot_age_confusion_matrices_per_population(df: pd.DataFrame):
    """Generuje macierze pomyłek dla predykcji wieku dla każdej populacji."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    populations = sorted(df['Populacja'].unique())

    if not populations:
        return None

    figs = []
    for pop in populations:
        pop_df = df[df['Populacja'] == pop]
        true_age = pop_df['Wiek'].astype(int)
        pred_age = pop_df['age_pred'].astype(int)

        labels = sorted(list(set(true_age) | set(pred_age)))

        cm = confusion_matrix(true_age, pred_age, labels=labels)

        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap='Greens', text_kw={'color': 'black'})

        # Podświetlenie przekątnej
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i == j:
                    disp.text_[i, j].set_weight('bold')
                    disp.text_[i, j].set_color('blue')

        ax.set_title(f'Macierz pomyłek wieku - Populacja {pop}')
        ax.set_xlabel('Wiek przewidziany (age_pred)')
        ax.set_ylabel('Wiek prawdziwy (Wiek)')
        plt.tight_layout()
        figs.append(fig)

    return figs


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
        cm_fig = plot_population_confusion_matrix(df_filtered)
        prob_dist_fig = plot_probability_distribution(df_filtered)
        age_scatter_fig = plot_age_scatter(df_filtered)
        prob_by_age_fig = plot_probability_by_age_class(df_filtered)
        age_cm_figs = plot_age_confusion_matrices_per_population(df_filtered)
        print("Wizualizacje wygenerowane.")

        # Zapis do pliku PDF
        pdf_path = log_dir / "prediction_analysis_report.pdf"
        print(f"\n🚀 Generowanie raportu PDF: {pdf_path}")
        with PdfPages(pdf_path) as pdf:
            if cm_fig:
                pdf.savefig(cm_fig)
            if prob_dist_fig:
                pdf.savefig(prob_dist_fig)
            if age_scatter_fig:
                pdf.savefig(age_scatter_fig)
            if prob_by_age_fig:
                pdf.savefig(prob_by_age_fig)
            if age_cm_figs:
                for fig in age_cm_figs:
                    pdf.savefig(fig)

        print(f"📄 Raport został pomyślnie zapisany.")
        plt.show()


    except Exception as e:
        print(f"Wystąpił błąd podczas przetwarzania pliku: {e}")


if __name__ == "__main__":
    main()
