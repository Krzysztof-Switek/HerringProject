import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.model import build_model
from utils.population_mapper import PopulationMapper
from omegaconf import OmegaConf
from pathlib import Path
from typing import Optional
from utils.heatmap_utils import create_heatmap_plot

# ================================================================
#                USTAWIENIA DO URUCHOMIENIA RĘCZNEGO
# ================================================================
MANUAL_RUN = True

if MANUAL_RUN:
    MANUAL_SETTINGS = {
        "model_path": "C:/Users/kswitek/Documents/HerringProject/checkpoints/BEST_resnet50_standard_ce_multi_2025-07-19_19-24/resnet50_standard_ce_SCORE_0.910.pth",
        "output_dir": "C:/Users/kswitek/Documents/HerringProject/results/logs/BEST_resnet50_standard_ce_multi_2025-07-19_19-24",
        "config_path": "src/config/config.yaml",
        "limit": None
    }


# ================================================================
#                         KONIEC SEKCJI
# ================================================================

def _resolve_local_image_path(row, data_root: Path) -> Optional[Path]:
    """
    Zwróć istniejącą ścieżkę do obrazu.
    1) Próbuj FilePath z DataFrame.
    2) Jeśli nie istnieje – szukaj po FileName w data_root (train/val/test/**).
    """
    # 1) spróbuj FilePath
    fp = row.get("FilePath", None)
    if isinstance(fp, str) and fp.strip():
        p = Path(fp)
        if p.exists():
            return p

    # 2) szukaj po FileName (case-insensitive) oraz rozszerzeniach
    fname = str(row.get("FileName", "")).strip()
    if not fname:
        return None

    candidates = []
    # dopasowanie bez uwzględniania wielkości liter + popularne rozszerzenia
    stem = Path(fname).stem
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    # przeszukaj train/val/test (płytko i rekurencyjnie)
    for split in ["train", "val", "test"]:
        base = data_root / split
        if not base.exists():
            continue
        # szukaj po nazwie bez rozszerzenia
        for ext in exts:
            for p in base.rglob(stem + ext):
                if p.name.lower() == (stem + ext).lower():
                    candidates.append(p)
    # wybierz pierwszy sensowny kandydat
    return candidates[0] if candidates else None


def generate_heatmaps_for_report(
        df_results: pd.DataFrame,
        model,
        cfg,
        log_dir: Path,
        transform,
        loss_name: str,
        data_root: Path,
        n_cases: int = 3
):
    """
    Generuje heatmapy dla trzech kategorii przypadków na podstawie *konkretnych*
    kolumn z bieżącego loss_name. Unika kolumn *_age_pred.
    """
    print("\n🔥 Generowanie heatmap dla raportu...")

    # --- 1) Wybór właściwych kolumn dokładnie po loss_name ---
    pred_col = f"{loss_name}_pred"
    prob_col = f"{loss_name}_prob"

    if pred_col not in df_results.columns or prob_col not in df_results.columns:
        # awaryjnie: spróbuj odfiltrować kolumny *_pred bez *_age_pred i wziąć najnowszą
        pred_candidates = [c for c in df_results.columns if c.endswith("_pred") and not c.endswith("_age_pred")]
        prob_candidates = [c for c in df_results.columns if c.endswith("_prob")]
        if not pred_candidates or not prob_candidates:
            print("⚠️ Nie znaleziono kolumn predykcji/prawdopodobieństwa dla heatmap. Pomijam generowanie.")
            return
        # weź ostatnie (najbardziej „najnowsze” w df)
        pred_col = pred_candidates[-1]
        prob_col = prob_candidates[-1]
        print(f"ℹ️ Uwaga: używam kolumn: pred={pred_col}, prob={prob_col}")

    # --- 2) Przygotowanie danych ---
    for col in [prob_col, pred_col, "Populacja"]:
        df_results[col] = pd.to_numeric(df_results[col], errors='coerce')

    # zachowaj tylko rekordy z kompletnymi danymi oraz istniejącym FileName
    base_subset = [prob_col, pred_col, "Populacja", "FileName"]
    keep_cols = [c for c in base_subset if c in df_results.columns]
    df_results = df_results.dropna(subset=keep_cols)

    # --- 3) Selekcja przypadków ---
    correct = df_results[df_results["Populacja"] == df_results[pred_col]]
    incorrect = df_results[df_results["Populacja"] != df_results[pred_col]]

    best_correct = correct.sort_values(by=prob_col, ascending=False).head(n_cases)
    worst_correct = correct.sort_values(by=prob_col, ascending=True).head(n_cases)
    confident_incorrect = incorrect.sort_values(by=prob_col, ascending=False).head(n_cases)

    print(f"   ✓ Znaleziono: best_correct={len(best_correct)}, worst_correct={len(worst_correct)}, "
          f"conf_incorrect={len(confident_incorrect)}")

    cases_to_plot = {
        "best_correct": best_correct,
        "worst_correct": worst_correct,
        "confident_incorrect": confident_incorrect
    }

    target_layer = cfg.visualization.target_layer
    device = next(model.parameters()).device

    heatmap_dir = log_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)

    # --- 4) Generowanie heatmap ---
    for category, df_cases in cases_to_plot.items():
        if df_cases.empty:
            print(f"INFO: Brak przypadków w kategorii '{category}'.")
            continue

        print(f"   -> Generowanie dla kategorii: {category}")
        category_dir = heatmap_dir / category
        category_dir.mkdir(exist_ok=True)

        for _, row in df_cases.iterrows():
            try:
                img_path = _resolve_local_image_path(row, data_root)
                if img_path is None or not img_path.exists():
                    print(f"    ❌ Nie znaleziono pliku lokalnie dla FileName='{row.get('FileName', '')}'. "
                          f"Pominięto.")
                    continue

                original_image = Image.open(img_path).convert("RGB")
                image_tensor = transform(original_image).unsqueeze(0).to(device)

                output_filename = f"{img_path.stem}.png"
                output_path = category_dir / output_filename

                create_heatmap_plot(
                    model=model,
                    target_layer=target_layer,
                    image_tensor=image_tensor,
                    original_image=original_image,
                    true_label=int(row["Populacja"]),
                    pred_label=int(row[pred_col]),
                    confidence=float(row[prob_col]),
                    output_path=str(output_path),
                    cfg_visualization=cfg.visualization
                )
            except Exception as e:
                print(f"    ❌ Błąd podczas generowania heatmapy: {e}")

    print("✅ Zakończono generowanie heatmap.")


def run_full_dataset_prediction(loss_name: str, model_path: str, path_manager,
                                log_dir, full_name: str, limit_predictions: Optional[int] = None):
    cfg = path_manager.cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    population_mapper = PopulationMapper(cfg.data.active_populations)

    # Build the model using the factory, which respects the 'mode' in the config
    model = build_model(cfg).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Redefine is_multitask based on the new config structure
    is_multitask = cfg.mode == 'multitask'

    image_size_to_use = cfg.image_size
    transform = transforms.Compose([
        transforms.Resize((image_size_to_use, image_size_to_use)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    excel_path = path_manager.metadata_file()
    df = pd.read_excel(excel_path)
    if "FilePath" not in df.columns:
        raise ValueError("Brakuje kolumny 'FilePath' zawierającej ścieżki do obrazów")

    data_root = path_manager.data_root()
    folders = [f"train/{pop}" for pop in population_mapper.active_populations] + \
              [f"val/{pop}" for pop in population_mapper.active_populations] + \
              [f"test/{pop}" for pop in population_mapper.active_populations]
    all_image_paths = [p for folder in folders for p in (data_root / folder).glob("*.jpg") if
                       (data_root / folder).exists()]
    all_image_paths.sort()

    if limit_predictions is not None and limit_predictions > 0:
        all_image_paths = all_image_paths[:limit_predictions]

    predictions = {}
    for i, image_path in enumerate(all_image_paths, 1):
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                if is_multitask:
                    pop_logits, age_pred = model(input_tensor)
                    probs = F.softmax(pop_logits, dim=1)[0]
                    pred_idx = pop_logits.argmax().item()
                    pred_class = population_mapper.to_pop(pred_idx)
                    confidence = float(probs[pred_idx]) * 100
                    predicted_age = float(age_pred[0].item())
                else:
                    pop_logits = model(input_tensor)
                    probs = F.softmax(pop_logits, dim=1)[0]
                    pred_idx = pop_logits.argmax().item()
                    pred_class = population_mapper.to_pop(pred_idx)
                    confidence = float(probs[pred_idx]) * 100
                    predicted_age = None
            key = image_path.name.lower()
            predictions[key] = (pred_class, round(confidence, 2), predicted_age)
        except Exception as e:
            print(f"Błąd przetwarzania obrazu {image_path}: {e}")

    pred_column = f"{loss_name}_pred"
    prob_column = f"{loss_name}_prob"
    age_column = f"{loss_name}_age_pred"

    pred_classes, pred_probs, pred_ages = [], [], []
    for file_name in df["FileName"]:
        pred = predictions.get(str(file_name).lower(), (None, None, None))
        pred_classes.append(pred[0])
        pred_probs.append(pred[1])
        pred_ages.append(pred[2])

    df[pred_column] = pred_classes
    df[prob_column] = pred_probs
    if is_multitask:
        df[age_column] = pred_ages

    output_path = log_dir / f"{full_name}_predictions.xlsx"
    df.to_excel(output_path, index=False)

    print(f"\n📊 Liczba przetworzonych obrazów: {len(predictions)}")
    print(f"✅ Zapisano predykcje ({loss_name}) do: {output_path}")

    if cfg.visualization.get('generate_heatmaps_in_report', False):
        generate_heatmaps_for_report(
            df_results=df,
            model=model,
            cfg=cfg,
            log_dir=log_dir,
            transform=transform,
            loss_name=loss_name,  # <— kluczowe: precyzyjny wybór kolumn
            data_root=path_manager.data_root(),  # <— do lokalnego wyszukania obrazów
            n_cases=cfg.visualization.get('n_heatmap_cases', 3)
        )

    print("⏭️ Przechodzę do kolejnej funkcji straty...\n")


if __name__ == '__main__':
    if MANUAL_RUN:
        from utils.path_manager import PathManager

        print("🚀 Uruchamianie predykcji w trybie ręcznym (z ustawień w pliku)...")
        settings = MANUAL_SETTINGS

        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / settings["config_path"]
        cfg = OmegaConf.load(config_path)

        path_manager = PathManager(project_root, cfg)
        output_dir = Path(settings["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = Path(settings["model_path"]).stem
        loss_name = f"manual_{model_name}"
        full_name = f"manual_run_{model_name}"

        run_full_dataset_prediction(
            loss_name=loss_name,
            model_path=settings["model_path"],
            path_manager=path_manager,
            log_dir=output_dir,
            full_name=full_name,
            limit_predictions=settings.get("limit")
        )
        print("✅ Predykcja w trybie ręcznym zakończona.")
    else:
        print("INFO: Aby uruchomić ten skrypt ręcznie, edytuj go i zmień flagę MANUAL_RUN на True.")