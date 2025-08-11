import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.model import HerringModel
from models.multitask_model import MultiTaskHerringModel
from utils.population_mapper import PopulationMapper
from omegaconf import OmegaConf
from pathlib import Path
from typing import Optional

# ================================================================
#                USTAWIENIA DO URUCHOMIENIA RĘCZNEGO
# ================================================================
# Ustaw MANUAL_RUN на True, aby uruchomić predykcję z poniższymi ustawieniami.
# Jeśli MANUAL_RUN jest False, ten blok jest ignorowany, a skrypt można
# importować i używać w pipeline treningowym.

MANUAL_RUN = False  # ZMIEŃ NA True, ABY URUCHOMIĆ RĘCZNIE

if MANUAL_RUN:
    MANUAL_SETTINGS = {
        "model_path": "checkpoints/nazwa_modelu/model.pth",
        "output_dir": "results/manual_prediction_output",
        "config_path": "src/config/config.yaml",  # Zazwyczaj nie trzeba zmieniać
        "limit": None  # Ustaw liczbę, np. 50, aby ograniczyć predykcje do testów
    }


# ================================================================
#                         KONIEC SEKCJI
# ================================================================


def run_full_dataset_prediction(loss_name: str, model_path: str, path_manager,
                                log_dir, full_name: str, limit_predictions: Optional[int] = None):
    import pandas as pd
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from models.model import HerringModel
    from models.multitask_model import MultiTaskHerringModel
    from utils.population_mapper import PopulationMapper

    if path_manager is None or path_manager.cfg is None:
        raise ValueError("path_manager (wraz z konfiguracją cfg) jest wymagany do uruchomienia predykcji.")

    cfg = path_manager.cfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    population_mapper = PopulationMapper(cfg.data.active_populations)

    is_multitask = getattr(cfg, "multitask_model", {}).get("use", False)
    if is_multitask:
        model = MultiTaskHerringModel(cfg).to(device)
    else:
        model = HerringModel(cfg).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    image_size_to_use = cfg.multitask_model.backbone_model.image_size if is_multitask else cfg.base_model.image_size
    transform = transforms.Compose([
        transforms.Resize(image_size_to_use),
        transforms.CenterCrop(image_size_to_use),
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
    all_image_paths = []
    for folder in folders:
        folder_path = data_root / folder
        if folder_path.exists():
            all_image_paths.extend(folder_path.glob("*.jpg"))

    all_image_paths.sort()

    if limit_predictions is not None and limit_predictions > 0:
        if len(all_image_paths) > limit_predictions:
            all_image_paths = all_image_paths[:limit_predictions]
            print(f"⚠️ Predykcje zostaną ograniczone do {limit_predictions} obrazów (tryb debug/testowy).")
        else:
            print(
                f"INFO: Żądano ograniczenia do {limit_predictions} obrazów, ale dostępnych jest tylko {len(all_image_paths)}. Użyte zostaną wszystkie dostępne.")

    predictions = {}
    total = len(all_image_paths)
    if total == 0:
        print(f"⚠️ Nie znaleziono obrazów do predykcji dla {loss_name}. Sprawdź ścieżki i konfigurację.")
        df_empty = pd.DataFrame(columns=['FileName', f"{loss_name}_pred", f"{loss_name}_prob"] + (
            [f"{loss_name}_age_pred"] if is_multitask else []))
        output_path = log_dir / f"{full_name}_predictions.xlsx"
        df_empty.to_excel(output_path, index=False)
        print(f"✅ Zapisano pusty plik predykcji ({loss_name}) do: {output_path}")
        return

    print(f"\n🔍 Start predykcji ({loss_name}) na {total} obrazach...")

    for i, image_path in enumerate(all_image_paths, 1):
        try:
            if total <= 200 or i % 100 == 0 or i == total:
                print(f"Przetworzono {i} z {total} obrazów...")
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
    pred_classes, pred_probs, pred_ages, not_found = [], [], [], []

    for file_name in df["FileName"]:
        key = str(file_name).lower()
        pred = predictions.get(key, (None, None, None))
        if pred[0] is None:
            not_found.append(key)
        pred_classes.append(pred[0])
        pred_probs.append(pred[1])
        pred_ages.append(pred[2])

    df[pred_column] = pred_classes
    df[prob_column] = pred_probs
    if is_multitask:
        df[age_column] = pred_ages

    output_path = log_dir / f"{full_name}_predictions.xlsx"
    if output_path.exists():
        existing_df = pd.read_excel(output_path)
        for col in [pred_column, prob_column, age_column]:
            if col in existing_df.columns:
                existing_df = existing_df.drop(columns=[col], errors='ignore')
        if is_multitask:
            df = pd.concat([existing_df, df[[pred_column, prob_column, age_column]]], axis=1)
        else:
            df = pd.concat([existing_df, df[[pred_column, prob_column]]], axis=1)

    df.to_excel(output_path, index=False)
    print(f"\n📊 Liczba przetworzonych obrazów: {len(predictions)}")
    print(f"✅ Zapisano predykcje ({loss_name}) do: {output_path}")
    print("⏭️ Przechodzę do kolejnej funkcji straty...\n")


if __name__ == '__main__':
    if MANUAL_RUN:
        # Importy potrzebne tylko do uruchomienia ręcznego
        from utils.path_manager import PathManager

        print("🚀 Uruchamianie predykcji w trybie ręcznym (z ustawień w pliku)...")

        settings = MANUAL_SETTINGS
        if not all(k in settings for k in ["model_path", "output_dir", "config_path"]):
            raise ValueError("MANUAL_SETTINGS musi zawierać 'model_path', 'output_dir', i 'config_path'.")

        # --- Setup ---
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / settings["config_path"]
        if not config_path.is_file():
            raise FileNotFoundError(f"Plik konfiguracyjny nie został znaleziony: {config_path}")

        cfg = OmegaConf.load(config_path)
        path_manager = PathManager(project_root, cfg)
        output_dir = Path(settings["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = Path(settings["model_path"]).stem
        loss_name = f"manual_{model_name}"
        full_name = f"manual_run_{model_name}"

        # Uruchom główną funkcję predykcji
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
