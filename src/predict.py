import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.model import HerringModel
from omegaconf import OmegaConf
import os

def run_full_dataset_prediction():
    # Wczytanie konfiguracji
    project_root = Path(__file__).parent.parent
    config_path = project_root / "src" / "config" / "config.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.data.root_dir = str((project_root / cfg.data.root_dir).resolve())
    cfg.prediction.model_path = str(project_root / cfg.prediction.model_path)
    model_name = cfg.model.base_model.replace("/", "_").lower()

    # Przygotowanie modelu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HerringModel(cfg).to(device)
    checkpoint = torch.load(cfg.prediction.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Transformacja
    transform = transforms.Compose([
        transforms.Resize(cfg.data.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Wczytanie Excela
    excel_path = project_root / "src" / "data_loader" / "AnalysisWithOtolithPhoto_with_sets.xlsx"
    df = pd.read_excel(excel_path)
    if "FilePath" not in df.columns:
        raise ValueError("Brakuje kolumny 'FilePath' zawierającej ścieżki do obrazów")

    # Lista folderów ze zdjęciami
    data_root = Path(cfg.data.root_dir)
    folders = ["train/1", "train/2", "val/1", "val/2", "test/1", "test/2"]
    all_image_paths = []
    for folder in folders:
        folder_path = data_root / folder
        if folder_path.exists():
            print(f"Przetwarzam katalog: {folder_path}")
            image_paths = list(folder_path.glob("*.jpg"))
            all_image_paths.extend(image_paths)
        else:
            print(f"[UWAGA] Pomijam brakujący katalog: {folder_path}")

    # Mapowanie ścieżek do predykcji
    predictions = {}
    for i, image_path in enumerate(all_image_paths, 1):
        try:
            if i % 100 == 0:
                print(f"Przetworzono {i} z {len(all_image_paths)} obrazów...")
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)[0]
                pred_class = output.argmax().item() +1
                confidence = float(probs[pred_class -1]) * 100
            key = image_path.name.lower()
            predictions[key] = (pred_class, round(confidence, 2))
        except Exception as e:
            print(f"Błąd przetwarzania obrazu {image_path}: {e}")

    print(f"\n📊 Liczba przetworzonych obrazów: {len(predictions)}")
    print(f"📁 Przykładowy klucz: {next(iter(predictions)) if predictions else 'brak danych'}")

    # Dopasowanie do pliku Excel
    pred_column = f"{model_name}_pred"
    prob_column = f"{model_name}_probability"
    pred_classes = []
    pred_probs = []
    not_found = []

    for file_name in df["FileName"]:
        file_name = str(file_name).lower()
        pred = predictions.get(file_name, (None, None))
        if pred[0] is None:
            not_found.append(file_name)
        pred_classes.append(pred[0])
        pred_probs.append(pred[1])

    print(f"\n❗Nie znaleziono predykcji dla {len(not_found)} z {len(df)} plików.")
    if not_found:
        print("Przykłady brakujących plików:", not_found[:5])

    df[pred_column] = pred_classes
    df[prob_column] = pred_probs

    # Zapis
    output_path = excel_path.parent / f"{excel_path.stem}_with_preds_{model_name}.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\n✅ Zapisano wyniki do: {output_path}")

if __name__ == "__main__":
    run_full_dataset_prediction()