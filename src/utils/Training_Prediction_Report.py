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
from .plots import generate_all_plots

matplotlib.rcParams.update(MATPLOTLIB_DEFAULTS)


class TrainingPredictionReport:
    def __init__(self, log_dir, base_config_obj, predictions_path, metadata_path,
                 run_params_obj=None, metrics_path=None, augmentation_path=None):
        self.log_dir = Path(log_dir)
        self.base_config = base_config_obj
        self.run_params = run_params_obj
        self.predictions_path = Path(predictions_path)
        self.metadata_path = Path(metadata_path)
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.augmentation_path = Path(augmentation_path) if augmentation_path else None
        self.metrics = None
        self.predictions = None
        self.metadata = None
        self.augmentation = None
        self.confusion_matrix = None

    def load_data(self):
        if self.metrics_path and self.metrics_path.exists():
            self.metrics = pd.read_csv(self.metrics_path)
        else:
            metrics_files = list(self.log_dir.glob("*training_metrics.csv"))
            self.metrics = pd.read_csv(metrics_files[0]) if metrics_files else None

        if self.metrics is not None:
            print(f"[DEBUG] Wczytane kolumny metryk: {self.metrics.columns.tolist()}")

        self.predictions = pd.read_excel(self.predictions_path)
        self.metadata = pd.read_excel(self.metadata_path)

        if self.augmentation_path and self.augmentation_path.exists():
            self.augmentation = pd.read_csv(self.augmentation_path)
        else:
            aug_files = list(self.log_dir.glob("augmentation_summary_*.csv"))
            self.augmentation = pd.read_csv(aug_files[0]) if aug_files else None

        cm_path_in_log = self.log_dir / "best_confusion_matrix.csv"
        cm_path_in_checkpoint = self.log_dir / self.log_dir.name / "best_confusion_matrix.csv"

        cm_path = None
        if cm_path_in_log.exists():
            cm_path = cm_path_in_log
        elif cm_path_in_checkpoint.exists():
            cm_path = cm_path_in_checkpoint

        if cm_path and cm_path.exists():
            try:
                self.confusion_matrix = pd.read_csv(cm_path, header=None).values
            except Exception as e:
                print(f"⚠️ Nie udało się wczytać macierzy pomyłek z {cm_path}: {e}")
        else:
            print("⚠️ Nie znaleziono pliku best_confusion_matrix.csv.")

    def extract_experiment_setup(self):
        cfg = self.base_config
        run_cfg = self.run_params
        setup = {}

        if run_cfg and hasattr(run_cfg, 'model_name_used'):
            setup["Model"] = run_cfg.model_name_used
        elif "base_model" in cfg and "base_model" in cfg["base_model"]:
            setup["Model"] = cfg["base_model"]["base_model"]
        elif "multitask_model" in cfg and "backbone_model" in cfg["multitask_model"]:
            setup["Model"] = cfg["multitask_model"]["backbone_model"]["model_name"]
        else:
            setup["Model"] = "NIEZNANY"

        if run_cfg and hasattr(run_cfg, 'model_mode'):
            mode_map = {"base": "klasyfikacja (base)", "multitask": "multitask", "expert": "expert"}
            setup["Typ modelu"] = mode_map.get(run_cfg.model_mode, run_cfg.model_mode)
        elif "multitask_model" in cfg and cfg.multitask_model.get("use", False):
            setup["Typ modelu"] = "multitask"
        else:
            setup["Typ modelu"] = "klasyfikacja (base)"

        if run_cfg and hasattr(run_cfg, 'loss_function_used'):
            setup["Funkcja straty (z params.yaml)"] = run_cfg.loss_function_used

        if run_cfg and hasattr(run_cfg, 'composite_score_weights'):
            weights = getattr(run_cfg, 'composite_score_weights', None)
            if weights and hasattr(weights, 'alpha'):
                setup["Wagi Composite Score (alpha)"] = getattr(weights, 'alpha', 'N/A')
                setup["Wagi Composite Score (beta)"] = getattr(weights, 'beta', 'N/A')
                setup["Wagi Composite Score (gamma)"] = getattr(weights, 'gamma', 'N/A')

        default_loss_from_base = "-"
        if hasattr(cfg, 'training') and cfg.training and hasattr(cfg.training, 'loss_type'):
            loss_type_val = cfg.training.loss_type
            default_loss_from_base = loss_type_val[0] if isinstance(loss_type_val, (list, tuple)) and len(
                loss_type_val) > 0 else str(loss_type_val)
        setup["Użyta funkcja straty"] = getattr(run_cfg, 'loss_function_used', default_loss_from_base)

        active_pops = cfg.data.get("active_populations", [])
        setup["Liczba klas populacji"] = len(active_pops)
        setup["Populacje (active_populations)"] = ", ".join(map(str, active_pops))

        max_age = cfg.get("data", {}).get("max_age") or cfg.get("multitask_model", {}).get("regression_head", {}).get(
            "max_age")
        setup["Liczba klas wiekowych"] = max_age if max_age else "brak/nie dotyczy"

        training_cfg = cfg.get("training", {})
        setup["batch_size"] = cfg.get("data", {}).get("batch_size", "nieznany")
        lr = training_cfg.get("learning_rate", "nieznany")
        setup["learning_rate"] = f"{float(lr):.6f}".replace('.', ',') if isinstance(lr, (float, str)) and "e" in str(
            lr) else str(lr)
        setup["optimizer"] = training_cfg.get("optimizer", "AdamW (domyślny)")

        if self.metrics is not None and not self.metrics.empty:
            real_epochs, max_epochs = len(self.metrics), training_cfg.get("epochs", "nieznany")
            setup[
                "epochs"] = f"{real_epochs} (zatrzymano przez early stopping, max={max_epochs})" if real_epochs < max_epochs else str(
                max_epochs)
        else:
            setup["epochs"] = training_cfg.get("epochs", "nieznany")

        aug = cfg.get("augmentation", {})
        setup["Augmentacja"] = "\n".join([f"{k}: {v}" for k, v in aug.items()]) or "Brak/nie ustawiono"
        self.experiment_setup = setup
        return setup

    def analyze(self):
        print("[REPORT] Analiza danych treningowych i predykcyjnych...")
        self.extract_experiment_setup()
        if self.metrics is not None and 'Val Composite Score' in self.metrics.columns:
            idx_best = self.metrics['Val Composite Score'].idxmax()
            best_row = self.metrics.loc[idx_best]
            self.best_metrics = {k: best_row.get(k) for k in self.metrics.columns}
            print(
                f"Znaleziono najlepszą epokę ({self.best_metrics['Epoch']}) z Composite Score: {self.best_metrics['Val Composite Score']:.4f}")
        else:
            self.best_metrics = None

    def add_heatmap_section(self, elements):
        """Finds and adds heatmap images to the report."""
        heatmap_dir = self.log_dir / "heatmaps"
        if not heatmap_dir.is_dir():
            return

        elements.append(ReportText("<b>Wizualizacja Modelu (Heatmaps)</b>", getSampleStyleSheet()['h2']))

        categories = {
            "best_correct": "Najlepsze Trafienia (Najwyższa Pewność)",
            "worst_correct": "Niepewne Trafienia (Najniższa Pewność)",
            "confident_incorrect": "Pewne Błędy (Najwyższa Pewność)"
        }

        for cat_dir, cat_title in categories.items():
            category_path = heatmap_dir / cat_dir
            if category_path.is_dir():
                images = sorted(list(category_path.glob("*.png")))
                if images:
                    elements.append(ReportText(cat_title, getSampleStyleSheet()['h3']))
                    for img_path in images:
                        elements.append(ReportImageRow([str(img_path)], height=REPORT_IMAGE_HEIGHT * 1.5))

    def generate_pdf(self):
        print("[REPORT] Generowanie PDF z podsumowaniem...")
        pdf_path = self.log_dir / f"Raport dla - {self.log_dir.name}.pdf"
        elements = []

        # --- Header ---
        elements.append(
            ReportText("<b>Podsumowanie treningu i predykcji</b>", getSampleStyleSheet()['Title'], spacer=12))
        elements.append(ReportText(
            f"Model: <b>{self.experiment_setup.get('Model', '-')}</b>, Typ: <b>{self.experiment_setup.get('Typ modelu', '-')}</b>"))
        elements.append(ReportText(f"Funkcja straty: <b>{self.experiment_setup.get('Użyta funkcja straty', '-')}</b>"))
        if self.metrics is not None and 'Val Composite Score' in self.metrics.columns:
            best_composite_score = self.metrics['Val Composite Score'].dropna().max()
            if not pd.isna(best_composite_score):
                elements.append(ReportText(f"Najlepszy Composite Score (val): <b>{best_composite_score:.3f}</b>"))

        # --- Training Params ---
        excluded_keys = ["Model", "Typ modelu", "Użyta funkcja straty", "Augmentacja", "Wagi Composite Score (alpha)",
                         "Wagi Composite Score (beta)", "Wagi Composite Score (gamma)",
                         "Funkcja straty (z params.yaml)"]
        params_data = [["Parametr", "Wartość"]] + [[k, v] for k, v in self.experiment_setup.items() if
                                                   k not in excluded_keys]
        elements.append(ReportTable(params_data))

        # --- Composite Score Weights ---
        if "Wagi Composite Score (alpha)" in self.experiment_setup:
            elements.append(ReportText("<b>Wagi Composite Score</b>", getSampleStyleSheet()['h2']))
            weights_data = [["alpha (F1 Global)", "beta (1 - MAE age)", "gamma (F1 Subgroup)"],
                            [f"{self.experiment_setup.get(k, 'N/A'):.2f}" for k in
                             ["Wagi Composite Score (alpha)", "Wagi Composite Score (beta)",
                              "Wagi Composite Score (gamma)"]]]
            elements.append(ReportTable(weights_data))

        # --- Best Metrics Table ---
        if self.best_metrics:
            elements.append(ReportText(
                f"<b>Szczegółowe metryki (dla najlepszej epoki: {self.best_metrics.get('Epoch', 'N/A')})</b>",
                getSampleStyleSheet()['h2']))
            val_metrics_data = [["Metryka walidacyjna", "Wartość"]] + [[k, f"{v:.4f}" if isinstance(v, float) else v]
                                                                       for k, v in self.best_metrics.items() if
                                                                       "Val" in k]
            elements.append(ReportTable(val_metrics_data))
            train_metrics_data = [["Metryka treningowa", "Wartość"]] + [[k, f"{v:.4f}" if isinstance(v, float) else v]
                                                                        for k, v in self.best_metrics.items() if
                                                                        "Train" in k and "Val" not in k]
            elements.append(ReportTable(train_metrics_data))

        # --- Plots ---
        elements.append(ReportText("<b>Wizualizacje treningu</b>", getSampleStyleSheet()['h2']))
        plot_files = generate_all_plots(self.metrics, self.confusion_matrix,
                                        [str(p) for p in self.base_config.data.active_populations], self.log_dir,
                                        self.predictions)
        for i in range(0, len(plot_files), 2):
            elements.append(ReportImageRow(plot_files[i:i + 2], height=REPORT_IMAGE_HEIGHT))

        # --- NEW: Heatmap Section ---
        self.add_heatmap_section(elements)

        # --- Augmentation Details ---
        if "Augmentacja" in self.experiment_setup and self.experiment_setup[
            "Augmentacja"].strip() != "Brak/nie ustawiono":
            elements.append(ReportText("<b>Parametry augmentacji</b>", getSampleStyleSheet()['h2']))
            elements.append(ReportText(self.experiment_setup["Augmentacja"].replace("\n", "<br/>")))

        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                                bottomMargin=MARGIN)
        doc.build([item for sublist in elements for item in sublist.get_flowables()])

        # Cleanup temporary plot files
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
