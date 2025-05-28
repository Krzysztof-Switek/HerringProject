import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages

# ============================
# 🔧 USTAWIENIA UŻYTKOWNIKA
log_dir = "resnet50_28-05"
# ============================

def load_metrics(log_dir):
    project_root = Path(__file__).resolve().parent.parent.parent
    metrics_path = project_root / "results" / "logs" / log_dir / "training_metrics.csv"
    cm_path = project_root / "results" / "logs" / log_dir / "confusion_matrix_best_model.npz"
    augment_path = project_root / "results" / "logs" / log_dir / "augment_usage_summary.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku metryk: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    confusion_matrix, labels = None, None
    if cm_path.exists():
        data = np.load(cm_path, allow_pickle=True)
        confusion_matrix = data['matrix']
        labels = data['labels'].tolist()
    else:
        print("⚠️ Nie znaleziono confusion_matrix_best_model.npz — nie zostanie wyświetlona.")

    augment_df = None
    if augment_path.exists():
        augment_df = pd.read_csv(augment_path)
    else:
        print("⚠️ Nie znaleziono augment_usage_summary.csv — nie zostanie uwzględniony.")

    return metrics, confusion_matrix, labels, augment_df

def plot_metrics(metrics):
    metrics['Epoch'] = metrics['Epoch'].astype(int)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    axes[0, 0].plot(metrics['Epoch'], metrics['Train Accuracy'], 'o-', label='Train')
    axes[0, 0].plot(metrics['Epoch'], metrics['Val Accuracy'], 'o-', label='Val')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(metrics['Epoch'], metrics['Train F1'], 'o-', label='Train')
    axes[0, 1].plot(metrics['Epoch'], metrics['Val F1'], 'o-', label='Val')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(metrics['Epoch'], metrics['Train Loss'], 'o-', label='Train')
    axes[1, 0].plot(metrics['Epoch'], metrics['Val Loss'], 'o-', label='Val')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(metrics['Epoch'], metrics['Train AUC'], 'o-', label='Train')
    axes[1, 1].plot(metrics['Epoch'], metrics['Val AUC'], 'o-', label='Val')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    axes[2, 0].plot(metrics['Epoch'], metrics['Train Precision'], 'o-', label='Train')
    axes[2, 0].plot(metrics['Epoch'], metrics['Val Precision'], 'o-', label='Val')
    axes[2, 0].set_ylabel('Precision')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    axes[2, 1].plot(metrics['Epoch'], metrics['Train Recall'], 'o-', label='Train')
    axes[2, 1].plot(metrics['Epoch'], metrics['Val Recall'], 'o-', label='Val')
    axes[2, 1].set_ylabel('Recall')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, labels):
    if cm is None:
        return None
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig

def create_summary_page(metrics):
    last_epoch = metrics['Epoch'].max()
    final_row = metrics[metrics['Epoch'] == last_epoch].iloc[0]
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    lines = [
        " PODSUMOWANIE TRENINGU ",
        "",
        f"Model: {log_dir.split('_')[0]}",
        f"Liczba epok: {last_epoch}",
        f"Val Accuracy: {final_row['Val Accuracy']:.2f}",
        f"Val F1: {final_row['Val F1']:.2f}",
        f"Val AUC: {final_row['Val AUC']:.2f}",
    ]

    if last_epoch < metrics.shape[0]:
        lines.append("\n⛔ Early stopping aktywowany")

    ax.text(0.05, 0.95, "\n".join(lines), fontsize=12, va='top', family='monospace')
    return fig

def main():
    metrics, confusion_matrix, labels, augment_df = load_metrics(log_dir)
    print(metrics.tail(3))

    fig1 = plot_metrics(metrics)
    fig2 = plot_confusion_matrix(confusion_matrix, labels)
    fig3 = create_summary_page(metrics)

    plt.show()

    export = input("Wygenerowa\u0107 raport PDF z metrykami i confusion matrix? (T/N): ").strip().lower()
    if export in ('', 't', 'tak', 'y', 'yes'):
        project_root = Path(__file__).resolve().parent.parent.parent
        pdf_path = project_root / "results" / "logs" / log_dir / "training_report.pdf"

        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig3)
            if fig1: pdf.savefig(fig1)
            if fig2: pdf.savefig(fig2)

            if augment_df is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(
                    augment_df.pivot(index='Wiek', columns='Populacja', values='AugmentacjaZastosowana').fillna(0),
                    annot=True, fmt=".0f", cmap="YlGnBu", cbar=False, ax=ax
                )
                ax.set_title("Zastosowanie augmentacji per klasa")
                plt.tight_layout()
                pdf.savefig(fig)

        print(f"📄 Raport zapisany do: {pdf_path}")

if __name__ == "__main__":
    main()
