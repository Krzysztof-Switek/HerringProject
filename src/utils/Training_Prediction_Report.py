import matplotlib.pyplot as plt
from .report_constants import MATPLOTLIB_DEFAULTS
import matplotlib
import pandas as pd
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet
from .report_elements import ReportText, ReportTable, ReportImageRow
from .report_constants import MARGIN, USABLE_WIDTH, REPORT_IMAGE_HEIGHT
from .plots import generate_report_plots
matplotlib.rcParams.update(MATPLOTLIB_DEFAULTS)


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
        # Model
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

        # Populacje
        active_pops = cfg["data"].get("active_populations", [])
        setup["Liczba klas populacji"] = len(active_pops)
        setup["Populacje (active_populations)"] = ", ".join(map(str, active_pops))

        # Klasy wiekowe
        max_age = None
        if "data" in cfg and "max_age" in cfg["data"]:
            max_age = cfg["data"]["max_age"]
        elif "multitask_model" in cfg and "regression_head" in cfg["multitask_model"]:
            max_age = cfg["multitask_model"]["regression_head"].get("max_age")
        setup["Liczba klas wiekowych"] = max_age if max_age else "brak/nie dotyczy"

        # Parametry treningu
        training_cfg = cfg.get("training", {})
        setup["batch_size"] = cfg.get("data", {}).get("batch_size", "nieznany")

        lr = training_cfg.get("learning_rate", "nieznany")
        if isinstance(lr, float):
            setup["learning_rate"] = f"{lr:.6f}".replace('.', ',')
        elif isinstance(lr, str) and "e" in lr:
            try:
                setup["learning_rate"] = f"{float(lr):.6f}".replace('.', ',')
            except Exception:
                setup["learning_rate"] = lr
        else:
            setup["learning_rate"] = str(lr)

        setup["optimizer"] = training_cfg.get("optimizer", "AdamW (domyślny)")

        # Epoki
        if self.metrics is not None and len(self.metrics) > 0:
            real_epochs = len(self.metrics)
            max_epochs = training_cfg.get("epochs", "nieznany")
            if real_epochs < max_epochs:
                setup["epochs"] = f"{real_epochs} (zatrzymano przez early stopping, max={max_epochs})"
            else:
                setup["epochs"] = str(max_epochs)
        else:
            setup["epochs"] = training_cfg.get("epochs", "nieznany")

        # Augmentacja
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
            idx_best = self.metrics["Val Accuracy"].idxmax()
            best_row = self.metrics.loc[idx_best]
            self.best_metrics = {
                "Val Accuracy": round(best_row["Val Accuracy"], 4),
                "Val F1": round(best_row["Val F1"], 4),
                "Val Recall": round(best_row["Val Recall"], 4),
                "Val Precision": round(best_row["Val Precision"], 4),
                "Val AUC": round(best_row["Val AUC"], 4),
            }
        else:
            self.best_metrics = None

    def _plot_placeholder(self, save_path, message="Brak danych!", width_inch=USABLE_WIDTH/25.4, height_inch=3):
        plt.figure(figsize=(width_inch, height_inch))
        plt.text(0.5, 0.5, message, ha='center', va='center', fontsize=18, color='red')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def generate_pdf(self):
        print("[REPORT] Generowanie PDF z podsumowaniem...")
        pdf_filename = f"Raport dla - {self.log_dir.name}.pdf"
        pdf_path = self.log_dir / pdf_filename
        elements = []

        elements.append(ReportText(f"<b>Podsumowanie eksperymentu</b>", getSampleStyleSheet()['Title'], spacer=12))
        elements.append(ReportText(f"Model: <b>{self.experiment_setup.get('Model', '-')}</b>"))
        elements.append(ReportText(f"Typ modelu: <b>{self.experiment_setup.get('Typ modelu', '-')}</b>"))
        loss_name = self.log_dir.name.split("_")[1] if "_" in self.log_dir.name else self.experiment_setup.get("loss_type", "-")
        elements.append(ReportText(f"Funkcja straty: <b>{loss_name}</b>"))
        acc_str = str(self.best_metrics["Val Accuracy"]) if self.best_metrics else "-"
        elements.append(ReportText(f"Najlepszy accuracy (val): <b>{acc_str}</b>"))

        elements.append(ReportText("<i>Parametry treningu:</i>", style=getSampleStyleSheet()['Italic']))
        for k, v in self.experiment_setup.items():
            if k not in ["Model", "Typ modelu", "loss_type", "Augmentacja"]:
                elements.append(ReportText(f"{k}: {v}"))

        # --- Tabela metryk treningowych ---
        if self.metrics is not None and self.best_metrics:
            idx_best = self.metrics["Val Accuracy"].idxmax()
            best_row = self.metrics.loc[idx_best]
            train_columns = [
                "Train Loss", "Train Accuracy", "Train Precision", "Train Recall", "Train F1", "Train AUC"
            ]
            train_values = [
                f"{best_row['Train Loss']:.2f}" if 'Train Loss' in best_row else "-",
                f"{best_row['Train Accuracy']:.2f}" if 'Train Accuracy' in best_row else "-",
                f"{best_row['Train Precision']:.2f}" if 'Train Precision' in best_row else "-",
                f"{best_row['Train Recall']:.2f}" if 'Train Recall' in best_row else "-",
                f"{best_row['Train F1']:.2f}" if 'Train F1' in best_row else "-",
                f"{best_row['Train AUC']:.2f}" if 'Train AUC' in best_row else "-",
            ]
            elements.append(ReportText("<b>Tabela metryk treningowych (najlepsza epoka)</b>", getSampleStyleSheet()['Title']))
            elements.append(ReportTable([train_columns, train_values]))

            val_columns = [
                "Val Loss", "Val Accuracy", "Val Precision", "Val Recall", "Val F1", "Val AUC"
            ]
            val_values = [
                f"{best_row['Val Loss']:.2f}" if 'Val Loss' in best_row else "-",
                f"{best_row['Val Accuracy']:.2f}" if 'Val Accuracy' in best_row else "-",
                f"{best_row['Val Precision']:.2f}" if 'Val Precision' in best_row else "-",
                f"{best_row['Val Recall']:.2f}" if 'Val Recall' in best_row else "-",
                f"{best_row['Val F1']:.2f}" if 'Val F1' in best_row else "-",
                f"{best_row['Val AUC']:.2f}" if 'Val AUC' in best_row else "-",
            ]
            elements.append(ReportText("<b>Tabela metryk walidacyjnych (najlepsza epoka)</b>", getSampleStyleSheet()['Title']))
            elements.append(ReportTable([val_columns, val_values]))

        # --- wykresy --- (AUTOMATYZACJA)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        plot_files = generate_report_plots(self.metrics, self.log_dir)
        elements.append(ReportText("<b>Wykresy metryk po epokach</b>", getSampleStyleSheet()['Title']))

        # Dodajemy wykresy parami w wierszu (np. dwa na wiersz)
        for i in range(0, len(plot_files), 2):
            pair = plot_files[i:i+2]
            elements.append(ReportImageRow(pair, height=REPORT_IMAGE_HEIGHT))

        if "Augmentacja" in self.experiment_setup and self.experiment_setup["Augmentacja"].strip() != "Brak/nie ustawiono":
            elements.append(ReportText("<b>Parametry augmentacji</b>", getSampleStyleSheet()['Title']))
            aug_lines = self.experiment_setup["Augmentacja"].split("\n")
            for line in aug_lines:
                elements.append(ReportText(line))

        story = []
        for elem in elements:
            story.extend(elem.get_flowables())

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN, bottomMargin=MARGIN
        )
        doc.build(story)

        # Usuwanie plików tymczasowych po wygenerowaniu PDF
        for plot_file in plot_files:
            try:
                Path(plot_file).unlink()
            except Exception:
                pass

        print(f"[REPORT] PDF zapisany: {pdf_path}")

    def run(self):
        print("[REPORT] Start generowania raportu PDF...")
        self.load_data()
        self.analyze()
        self.generate_pdf()
        print("[REPORT] Raport PDF wygenerowany w:", self.log_dir)
