import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.model import HerringModel

def run_full_dataset_prediction(loss_name: str, model_path: str, path_manager):
    cfg = path_manager.cfg

    # Przygotowanie modelu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HerringModel(cfg).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Transformacja obrazu
    transform = transforms.Compose([
        transforms.Resize(cfg.data.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Wczytanie Excela
    excel_path = path_manager.metadata_file()
    df = pd.read_excel(excel_path)
    if "FilePath" not in df.columns:
        raise ValueError("Brakuje kolumny 'FilePath' zawierającej ścieżki do obrazów")

    # Lista folderów
    data_root = path_manager.data_root()
    folders = ["train/1", "train/2", "val/1", "val/2", "test/1", "test/2"]
    all_image_paths = []
    for folder in folders:
        folder_path = data_root / folder
        if folder_path.exists():
            all_image_paths.extend(folder_path.glob("*.jpg"))

    # Predykcje
    predictions = {}
    for image_path in all_image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)[0]
                pred_class = output.argmax().item() + 1
                confidence = float(probs[pred_class - 1]) * 100
            key = image_path.name.lower()
            predictions[key] = (pred_class, round(confidence, 2))
        except Exception as e:
            print(f"Błąd przetwarzania obrazu {image_path}: {e}")

    # Kolumny predykcji
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

    # Zapis
    output_path = path_manager.excel_predictions_output()
    if output_path.exists():
        existing_df = pd.read_excel(output_path)
        existing_df = existing_df.drop(columns=[pred_column, prob_column], errors='ignore')
        df = pd.concat([existing_df, df[[pred_column, prob_column]]], axis=1)

    df.to_excel(output_path, index=False)
    print(f"\n✅ Zapisano predykcje ({loss_name}) do: {output_path}")
