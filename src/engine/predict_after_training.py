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

    # Konwersja log_dir do Path, jeśli jest stringiem
    log_dir_path = Path(log_dir)
    params_file = log_dir_path / "params.yaml"

    if params_file.exists():
        print(f"Ładowanie konfiguracji z pliku: {params_file}")
        cfg = OmegaConf.load(params_file)
    else:
        print(f"OSTRZEŻENIE: Plik konfiguracyjny {params_file} nie znaleziony. Używam konfiguracji z path_manager.")
        if path_manager is None or path_manager.cfg is None:
            raise ValueError("Nie można załadować konfiguracji: params.yaml nie istnieje, a path_manager lub path_manager.cfg jest None.")
        cfg = path_manager.cfg

    # PathManager może nadal być potrzebny do uzyskania project_root, jeśli ścieżki w cfg są względne
    # lub do innych zadań. Jeśli path_manager jest None, a potrzebny, to błąd.
    if path_manager is None and not params_file.exists():
        # Jeśli nie ma params.yaml i nie ma path_manager, to nie mamy ani cfg, ani sposobu na ścieżki.
        raise ValueError("Nie można załadować konfiguracji ani uzyskać ścieżek: params.yaml nie istnieje, a path_manager jest None.")

    if path_manager is None and params_file.exists():
        # Mamy cfg z params.yaml, ale nie mamy project_root do utworzenia nowego PathManagera.
        # W tej sytuacji zakładamy, że ścieżki w params.yaml są absolutne lub praca bez PathManagera jest niemożliwa.
        # To jest mało prawdopodobny scenariusz, jeśli funkcja jest wywoływana z trainer_setup.
        # Dla bezpieczeństwa, rzućmy błąd, jeśli path_manager jest potrzebny do określenia project_root.
        # Można by zmodyfikować PathManager, aby nie wymagał project_root, jeśli ścieżki w cfg są absolutne.
        # Lub params.yaml musiałby przechowywać project_root.
        # Na razie: jeśli params.yaml istnieje, ale path_manager nie, to jest to problem dla PathManagera.
        # Chyba że cfg z params.yaml zawiera wszystkie potrzebne ścieżki jako absolutne.
        print("OSTRZEŻENIE: path_manager jest None. Ścieżki w załadowanej konfiguracji cfg muszą być absolutne lub poprawnie skonfigurowane.")
        # W tej sytuacji, jeśli cfg.data.root_dir i cfg.data.metadata_file są np. względne, to PathManager by ich nie rozwiązał poprawnie.
        # Dla uproszczenia, na razie zakładamy, że jeśli path_manager jest None, to cfg musi zawierać absolutne ścieżki.
        # Lepszym rozwiązaniem byłoby przekazanie project_root jako argumentu.
        # Na razie, jeśli path_manager jest None, nie tworzymy nowego. To spowoduje błąd później, jeśli metody path_manager będą wywołane.
        # To jest złe. PathManager jest potrzebny.
        raise ValueError("path_manager jest None. Jest on wymagany do działania tej funkcji.")


    # Utwórz nową instancję PathManager z załadowaną konfiguracją cfg
    # Potrzebujemy project_root z oryginalnego path_manager
    # To jest kluczowe: oryginalny path_manager dostarcza project_root.
    from utils.path_manager import PathManager as PM_local # Uniknięcie konfliktu nazw, jeśli PathManager byłby też tu importowany
    current_pm = PM_local(project_root=path_manager.project_root, cfg=cfg)

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

    excel_path = current_pm.metadata_file() # Użyj current_pm
    df = pd.read_excel(excel_path)
    if "FilePath" not in df.columns: # Pozostaje bez zmian
        raise ValueError("Brakuje kolumny 'FilePath' zawierającej ścieżki do obrazów")

    data_root = current_pm.data_root() # Użyj current_pm
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

