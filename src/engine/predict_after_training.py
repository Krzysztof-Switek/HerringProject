import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.model import HerringModel
from models.multitask_model import MultiTaskHerringModel
from utils.population_mapper import PopulationMapper  # 🟢 DODANO
from omegaconf import OmegaConf # <-- Dodano import
from pathlib import Path # <-- Dodano import

def run_full_dataset_prediction(loss_name: str, model_path: str, path_manager, log_dir, full_name):
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

    # Używamy konfiguracji (cfg) przekazanej przez path_manager.
    # Ta cfg to efektywna konfiguracja z danego przebiegu treningowego.
    # Minimalistyczny params.yaml nie jest tutaj używany do odtwarzania cfg.
    cfg = path_manager.cfg

    # Oryginalny path_manager jest używany bezpośrednio, ponieważ zawiera właściwą cfg.
    # Nie ma potrzeby tworzenia current_pm.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    population_mapper = PopulationMapper(cfg.data.active_populations)

    # Ustal typ modelu
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

    # Użyj cfg (załadowanego z params.yaml lub fallback) do określenia image_size
    image_size_to_use = cfg.multitask_model.backbone_model.image_size if is_multitask else cfg.base_model.image_size
    transform = transforms.Compose([
        transforms.Resize(image_size_to_use), # Użyj poprawnego image_size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    excel_path = path_manager.metadata_file() # Użyj oryginalnego path_manager
    df = pd.read_excel(excel_path)
    if "FilePath" not in df.columns: # Pozostaje bez zmian
        raise ValueError("Brakuje kolumny 'FilePath' zawierającej ścieżki do obrazów")

    data_root = path_manager.data_root() # Użyj oryginalnego path_manager
    folders = [f"train/{pop}" for pop in population_mapper.active_populations] + \
              [f"val/{pop}" for pop in population_mapper.active_populations] + \
              [f"test/{pop}" for pop in population_mapper.active_populations]
    all_image_paths = []
    for folder in folders:
        folder_path = data_root / folder
        if folder_path.exists():
            all_image_paths.extend(folder_path.glob("*.jpg"))

    predictions = {}
    total = len(all_image_paths)
    print(f"\n🔍 Start predykcji ({loss_name}) na {total} obrazach...")

    for i, image_path in enumerate(all_image_paths, 1):
        try:
            if i % 100 == 0 or i == total:
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
                    predicted_age = float(age_pred[0].item())  # <-- WIEK jako liczba zmiennoprzecinkowa
                else:
                    pop_logits = model(input_tensor)
                    probs = F.softmax(pop_logits, dim=1)[0]
                    pred_idx = pop_logits.argmax().item()
                    pred_class = population_mapper.to_pop(pred_idx)
                    confidence = float(probs[pred_idx]) * 100
                    predicted_age = None  # brak predykcji wieku
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
        # Dodajemy tylko nowe kolumny
        if is_multitask:
            df = pd.concat([existing_df, df[[pred_column, prob_column, age_column]]], axis=1)
        else:
            df = pd.concat([existing_df, df[[pred_column, prob_column]]], axis=1)

    df.to_excel(output_path, index=False)
    print(f"\n📊 Liczba przetworzonych obrazów: {len(predictions)}")
    print(f"✅ Zapisano predykcje ({loss_name}) do: {output_path}")
    print("⏭️ Przechodzę do kolejnej funkcji straty...\n")

