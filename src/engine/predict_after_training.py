import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.model import HerringModel
from models.multitask_model import MultiTaskHerringModel
from utils.population_mapper import PopulationMapper  # 🟢 DODANO

def run_full_dataset_prediction(loss_name: str, model_path: str, path_manager, log_dir, full_name):
    cfg = path_manager.cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🟢 PopulationMapper zgodnie z configiem
    population_mapper = PopulationMapper(cfg.data.active_populations)  # 🟢 DODANO

    # Wybór odpowiedniego modelu
    if getattr(cfg, "multitask_model", {}).get("use", False):
        model = MultiTaskHerringModel(cfg).to(device)
        is_multitask = True
    else:
        model = HerringModel(cfg).to(device)
        is_multitask = False

    # Wczytanie wag
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Transformacja
    transform = transforms.Compose([
        transforms.Resize(cfg.base_model.image_size),
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
              [f"test/{pop}" for pop in population_mapper.active_populations]   # 🟢 UOGÓLNIENIE folderów
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
                output = model(input_tensor)
                if is_multitask:
                    output = output[0]  # klasyfikacja (pomijamy predykcję wieku)
                probs = F.softmax(output, dim=1)[0]
                pred_idx = output.argmax().item()  # 🟢 INDEKS modelu (np. 0, 1)
                pred_class = population_mapper.to_pop(pred_idx)  # 🟢 NUMER populacji z Excela!
                confidence = float(probs[pred_idx]) * 100
            key = image_path.name.lower()
            predictions[key] = (pred_class, round(confidence, 2))
        except Exception as e:
            print(f"Błąd przetwarzania obrazu {image_path}: {e}")

    # Zapis predykcji
    pred_column = f"{loss_name}_pred"
    prob_column = f"{loss_name}_prob"
    pred_classes, pred_probs, not_found = [], [], []

    for file_name in df["FileName"]:
        key = str(file_name).lower()
        pred = predictions.get(key, (None, None))
        if pred[0] is None:
            not_found.append(key)
        pred_classes.append(pred[0])
        pred_probs.append(pred[1])

    df[pred_column] = pred_classes
    df[prob_column] = pred_probs

    output_path = log_dir / f"{full_name}_predictions.xlsx"
    if output_path.exists():
        existing_df = pd.read_excel(output_path)
        existing_df = existing_df.drop(columns=[pred_column, prob_column], errors='ignore')
        df = pd.concat([existing_df, df[[pred_column, prob_column]]], axis=1)

    df.to_excel(output_path, index=False)
    print(f"\n📊 Liczba przetworzonych obrazów: {len(predictions)}")
    print(f"✅ Zapisano predykcje ({loss_name}) do: {output_path}")
    print("⏭️ Przechodzę do kolejnej funkcji straty...\n")
