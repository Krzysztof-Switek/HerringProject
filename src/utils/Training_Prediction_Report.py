import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class TrainingPredictionReport:
    def __init__(self, log_dir, config_path, predictions_path, metadata_path,
                 metrics_path=None, augmentation_path=None):
        self.log_dir = Path(log_dir)
        self.config_path = Path(config_path)
        self.predictions_path = Path(predictions_path)
        self.metadata_path = Path(metadata_path)
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.augmentation_path = Path(augmentation_path) if augmentation_path else None

        self.metrics = None
        self.predictions = None
        self.metadata = None
        self.augmentation = None
        self.config = None

    def load_data(self):
        if self.metrics_path and self.metrics_path.exists():
            self.metrics = pd.read_csv(self.metrics_path)
        else:
            metrics_files = list(self.log_dir.glob("*training_metrics.csv"))
            self.metrics = pd.read_csv(metrics_files[0]) if metrics_files else None

        self.predictions = pd.read_excel(self.predictions_path)
        self.metadata = pd.read_excel(self.metadata_path)

        if self.augmentation_path and self.augmentation_path.exists():
            self.augmentation = pd.read_csv(self.augmentation_path)
        else:
            aug_files = list(self.log_dir.glob("augmentation_summary_*.csv"))
            self.augmentation = pd.read_csv(aug_files[0]) if aug_files else None

        import yaml
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def extract_experiment_setup(self):
        cfg = self.config
        setup = {}

        # Model i backbone
        if "base_model" in cfg and "base_model" in cfg["base_model"]:
            setup["Model"] = cfg["base_model"]["base_model"]
        elif "multitask_model" in cfg and "backbone_model" in cfg["multitask_model"]:
            setup["Model"] = cfg["multitask_model"]["backbone_model"]["model_name"]
        else:
            setup["Model"] = "NIEZNANY"

        # Typ modelu
        setup["Typ modelu"] = (
            "multitask"
            if "multitask_model" in cfg and cfg["multitask_model"].get("use", False)
            else "klasyfikacja"
        )

        # Liczba klas populacji i lista populacji
        active_pops = cfg["data"].get("active_populations", [])
        setup["Liczba klas populacji"] = len(active_pops)
        setup["Populacje (active_populations)"] = ", ".join(map(str, active_pops))

        # Liczba klas wiekowych
        max_age = None
        if "data" in cfg and "max_age" in cfg["data"]:
            max_age = cfg["data"]["max_age"]
        elif "multitask_model" in cfg and "regression_head" in cfg["multitask_model"]:
            max_age = cfg["multitask_model"]["regression_head"].get("max_age")
        setup["Liczba klas wiekowych"] = max_age if max_age else "brak/nie dotyczy"

        # Parametry treningu
        training_cfg = cfg.get("training", {})
        # --- batch_size: z data ---
        setup["batch_size"] = cfg.get("data", {}).get("batch_size", "nieznany")
        setup["learning_rate"] = training_cfg.get("learning_rate", "nieznany")
        setup["optimizer"] = training_cfg.get("optimizer", "AdamW (domyślny)")
        # --- epochs: z metrics.csv jeśli jest ---
        if self.metrics is not None and len(self.metrics) > 0:
            real_epochs = len(self.metrics)
            max_epochs = training_cfg.get("epochs", "nieznany")
            if real_epochs < max_epochs:
                setup["epochs"] = f"{real_epochs} (zatrzymano przez early stopping, max={max_epochs})"
            else:
                setup["epochs"] = str(max_epochs)
        else:
            setup["epochs"] = training_cfg.get("epochs", "nieznany")

        # Ustawienia augmentacji
        aug = cfg.get("augmentation", {})
        augment_info = []
        for k, v in aug.items():
            augment_info.append(f"{k}: {v}")
        setup["Augmentacja"] = "\n".join(augment_info) if augment_info else "Brak/nie ustawiono"

        self.experiment_setup = setup
        return setup

    def analyze(self):
        print("[REPORT] Analiza danych treningowych i predykcyjnych...")

        self.extract_experiment_setup()
        if self.metrics is not None:
            # Znajdź NAJLEPSZY wiersz po indeksie max Val Accuracy
            idx_best = self.metrics["Val Accuracy"].idxmax()
            best_row = self.metrics.loc[idx_best]
            # Wyciągnij string z kolumny "Epoch"
            best_epoch_label = best_row["Epoch"]
            self.best_metrics = {
                "Val Accuracy": round(best_row["Val Accuracy"], 4),
                "Val F1": round(best_row["Val F1"], 4),
                "Val Recall": round(best_row["Val Recall"], 4),
                "Val Precision": round(best_row["Val Precision"], 4),
                "Val AUC": round(best_row["Val AUC"], 4),
                "Epoch": best_epoch_label  # <-- teraz string np. "standard_ce-e5"
            }
        else:
            self.best_metrics = None

    def generate_pdf(self):
        print("[REPORT] Generowanie PDF z podsumowaniem...")

        # Nazwa pliku PDF = nazwa katalogu .pdf
        pdf_filename = f"Raport dla - {self.log_dir.name}.pdf"
        pdf_path = self.log_dir / pdf_filename

        with PdfPages(pdf_path) as pdf:
            # --- STRONA 1: info główne + metryki (tekstowo) ---
            fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
            ax.axis("off")
            y = 0.97
            ax.text(0.05, y, f"Model: {self.experiment_setup.get('Model', '-')}", fontsize=20, fontweight='bold',
                    transform=ax.transAxes)
            y -= 0.055
            ax.text(0.05, y, f"Typ modelu: {self.experiment_setup.get('Typ modelu', '-')}", fontsize=14,
                    fontweight='bold', transform=ax.transAxes)
            y -= 0.04
            loss_name = self.log_dir.name.split("_")[1] if "_" in self.log_dir.name else self.experiment_setup.get(
                "loss_type", "-")
            ax.text(0.05, y, f"Funkcja straty: {loss_name}", fontsize=14, fontweight='bold', transform=ax.transAxes)
            y -= 0.04
            acc_str = str(self.best_metrics["Val Accuracy"]) if self.best_metrics else "-"
            ax.text(0.05, y, f"Najlepszy accuracy (val): {acc_str}", fontsize=14, fontweight='bold',
                    transform=ax.transAxes)
            y -= 0.045

            # --- WYPISZ METRYKI JAKO TEKST ---
            if self.best_metrics:
                ax.text(0.05, y, f"Metryki najlepszej epoki:", fontsize=12, fontweight="bold", transform=ax.transAxes)
                y -= 0.03
                ax.text(0.07, y, f"- Epoka: {self.best_metrics['Epoch']}", fontsize=11, transform=ax.transAxes)
                y -= 0.024
                ax.text(0.07, y, f"- F1: {self.best_metrics['Val F1']}", fontsize=11, transform=ax.transAxes)
                y -= 0.024
                ax.text(0.07, y, f"- Recall: {self.best_metrics['Val Recall']}", fontsize=11, transform=ax.transAxes)
                y -= 0.024
                ax.text(0.07, y, f"- Precision: {self.best_metrics['Val Precision']}", fontsize=11,
                        transform=ax.transAxes)
                y -= 0.024
                ax.text(0.07, y, f"- AUC: {self.best_metrics['Val AUC']}", fontsize=11, transform=ax.transAxes)
                y -= 0.024

                # --- POPRAWKA: Pobierz metryki z właściwego wiersza ---
                row = self.metrics[self.metrics["Epoch"] == self.best_metrics['Epoch']]
                if len(row) > 0:
                    train_acc = row.iloc[0]['Train Accuracy']
                    train_loss = row.iloc[0]['Train Loss']
                    val_loss = row.iloc[0]['Val Loss']
                    ax.text(0.07, y, f"- Train accuracy: {train_acc:.4f}", fontsize=11, transform=ax.transAxes)
                    y -= 0.024
                    ax.text(0.07, y, f"- Train loss: {train_loss:.4f}", fontsize=11, transform=ax.transAxes)
                    y -= 0.024
                    ax.text(0.07, y, f"- Val loss: {val_loss:.4f}", fontsize=11, transform=ax.transAxes)
                    y -= 0.03

            # --- PARAMETRY TRENINGU ---
            ax.text(0.05, y, "Parametry treningu:", fontsize=12, fontweight="bold", transform=ax.transAxes)
            y -= 0.028
            for k, v in self.experiment_setup.items():
                if k not in ["Model", "Typ modelu", "loss_type", "Augmentacja"]:
                    lines = str(v).split("\n")
                    for idx, line in enumerate(lines):
                        prefix = f"{k}:" if idx == 0 else ""
                        ax.text(0.07, y, f"{prefix} {line}", fontsize=10, transform=ax.transAxes)
                        y -= 0.021
                    if len(lines) > 1:
                        y -= 0.006
            pdf.savefig(fig)
            plt.close(fig)

            # --- STRONA 2: WYKRESY ---
            if self.metrics is not None:
                fig3, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))  # Pionowo A4!
                fig3.subplots_adjust(hspace=0.4)
                epochs = list(range(1, len(self.metrics) + 1))
                axes[0].plot(epochs, self.metrics["Train Loss"], label="Train Loss", marker='o')
                axes[0].plot(epochs, self.metrics["Val Loss"], label="Val Loss", marker='o')
                axes[0].set_ylabel("Loss")
                axes[0].set_xlabel("Epoka")
                axes[0].set_title("Loss (trening/validacja)")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                axes[1].plot(epochs, self.metrics["Train Accuracy"], label="Train Acc", marker='o')
                axes[1].plot(epochs, self.metrics["Val Accuracy"], label="Val Acc", marker='o')
                axes[1].set_ylabel("Accuracy (%)")
                axes[1].set_xlabel("Epoka")
                axes[1].set_title("Accuracy (trening/validacja)")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                pdf.savefig(fig3)
                plt.close(fig3)

            # --- STRONA 3: Augmentacje (osobno, tylko jeśli istnieją) ---
            if "Augmentacja" in self.experiment_setup and self.experiment_setup[
                "Augmentacja"].strip() != "Brak/nie ustawiono":
                fig_aug, ax_aug = plt.subplots(figsize=(8.27, 11.69))
                ax_aug.axis("off")
                ax_aug.text(0.05, 0.95, "Parametry augmentacji:", fontsize=13, fontweight="bold",
                            transform=ax_aug.transAxes)
                aug_lines = self.experiment_setup["Augmentacja"].split("\n")
                y_aug = 0.91
                for line in aug_lines:
                    ax_aug.text(0.07, y_aug, line, fontsize=11, transform=ax_aug.transAxes)
                    y_aug -= 0.023
                pdf.savefig(fig_aug)
                plt.close(fig_aug)

        print(f"[REPORT] PDF zapisany: {pdf_path}")

    def _pdf_main_info(self, pdf):
        """Pierwsza strona: główne informacje o eksperymencie."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        y = 0.93
        ax.text(0.05, y, f"Model: {self.experiment_setup.get('Model', '-')}", fontsize=22, fontweight='bold', transform=ax.transAxes)
        y -= 0.07
        ax.text(0.05, y, f"Typ modelu: {self.experiment_setup.get('Typ modelu', '-')}", fontsize=16, fontweight='bold', transform=ax.transAxes)
        y -= 0.06
        loss_name = self.log_dir.name.split("_")[1] if "_" in self.log_dir.name else self.experiment_setup.get("loss_type", "-")
        ax.text(0.05, y, f"Funkcja straty: {loss_name}", fontsize=16, fontweight='bold', transform=ax.transAxes)
        y -= 0.06
        acc_str = str(self.best_metrics["Val Accuracy"]) if self.best_metrics else "-"
        ax.text(0.05, y, f"Najlepszy accuracy (val): {acc_str}", fontsize=16, fontweight='bold', transform=ax.transAxes)
        y -= 0.06
        if self.best_metrics:
            ax.text(0.05, y, f"F1: {self.best_metrics['Val F1']}, Recall: {self.best_metrics['Val Recall']}, "
                             f"Precision: {self.best_metrics['Val Precision']}, AUC: {self.best_metrics['Val AUC']}",
                    fontsize=13, transform=ax.transAxes)
            y -= 0.045
            ax.text(0.05, y, f"Liczba epok: {self.best_metrics['Epoch']} (z early stopping jeśli dotyczy)",
                    fontsize=13, transform=ax.transAxes)
            y -= 0.045
        ax.text(0.05, y, "Parametry treningu i augmentacji:", fontsize=13, fontweight="bold", transform=ax.transAxes)
        y -= 0.045
        for k, v in self.experiment_setup.items():
            if k not in ["Model", "Typ modelu", "loss_type"]:
                lines = str(v).split("\n")
                for idx, line in enumerate(lines):
                    prefix = f"{k}:" if idx == 0 else ""
                    ax.text(0.06, y, f"{prefix} {line}", fontsize=11, transform=ax.transAxes)
                    y -= 0.028
                if len(lines) > 1:
                    y -= 0.01
        pdf.savefig(fig)
        plt.close(fig)

    def _pdf_metrics_table(self, pdf):
        """Druga strona: czytelna tabela metryk najlepszej epoki (układ poziomy, większa czcionka)."""
        if self.metrics is not None and self.best_metrics:
            fig2, ax2 = plt.subplots(figsize=(11.7, 3.5))  # A4 szerokość, mniej wysokości
            ax2.axis("off")
            ax2.set_title("Tabela metryk walidacyjnych (najlepsza epoka)", fontsize=15, fontweight="bold", pad=18)
            best_epoch_label = self.best_metrics['Epoch']
            # Jeśli best_epoch_label to np. "standard_ce-e5", wyciągnij numer epoki z końca
            try:
                epoch_number = int(str(best_epoch_label).split("-e")[-1])
            except Exception:
                epoch_number = best_epoch_label
            row = self.metrics[self.metrics["Epoch"] == best_epoch_label]
            columns = [
                "Epoch", "Train Loss", "Train Accuracy", "Train F1",
                "Val Loss", "Val Accuracy", "Val F1", "Val Precision", "Val Recall", "Val AUC"
            ]
            if len(row) == 0:
                data = [[self.best_metrics.get(col, "-") for col in columns]]
            else:
                values = [row.iloc[0][c] if c in row else "-" for c in columns]
                # Zamień Epoch na czytelny numer epoki
                values[0] = epoch_number
                data = [values]
            table = ax2.table(
                cellText=data,
                colLabels=columns,
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.25, 2.2)
            # Podkreśl nagłówek
            for (row_i, col_i), cell in table.get_celld().items():
                if row_i == 0: cell.set_fontsize(13); cell.set_text_props(weight='bold')
                cell.set_height(0.2)
            plt.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)

    def _pdf_metrics_plots(self, pdf):
        """Trzecia strona: wykresy metryk po epokach w układzie do druku A4."""
        if self.metrics is not None:
            fig3, axes = plt.subplots(2, 1, figsize=(11.7, 8.3))  # A4 poziomo

            # Ekstraktuj numer epoki (np. z 'standard_ce-e5' -> 5)
            def epoch_num(e):
                try:
                    return int(str(e).split("-e")[-1])
                except Exception:
                    return str(e)

            epochs = [epoch_num(e) for e in self.metrics["Epoch"]]

            axes[0].plot(epochs, self.metrics["Train Loss"], label="Train Loss", linewidth=2)
            axes[0].plot(epochs, self.metrics["Val Loss"], label="Val Loss", linewidth=2)
            axes[0].set_ylabel("Loss", fontsize=12)
            axes[0].set_xlabel("Epoka", fontsize=12)
            axes[0].set_title("Loss (trening/validacja)", fontsize=14)
            axes[0].legend(fontsize=11)
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis='x', labelsize=11)
            axes[0].set_xticks(epochs)

            axes[1].plot(epochs, self.metrics["Train Accuracy"], label="Train Acc", linewidth=2)
            axes[1].plot(epochs, self.metrics["Val Accuracy"], label="Val Acc", linewidth=2)
            axes[1].set_ylabel("Accuracy (%)", fontsize=12)
            axes[1].set_xlabel("Epoka", fontsize=12)
            axes[1].set_title("Accuracy (trening/validacja)", fontsize=14)
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(axis='x', labelsize=11)
            axes[1].set_xticks(epochs)

            plt.tight_layout()
            pdf.savefig(fig3)
            plt.close(fig3)


    def run(self):
        print("[REPORT] Start generowania raportu PDF...")
        self.load_data()
        self.analyze()
        self.generate_pdf()
        print("[REPORT] Raport PDF wygenerowany w:", self.log_dir)
