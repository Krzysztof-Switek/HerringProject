from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from ..data_loader.transforms import get_eval_transform
from ..models.checkpoint_utils import load_model_checkpoint
from ..models.model import build_model
from ..utils.population_mapper import PopulationMapper


def _collect_all_image_paths(data_root: Path, active_populations) -> list[Path]:
    image_paths: list[Path] = []

    for split in ("train", "val", "test"):
        for pop in active_populations:
            folder = data_root / split / str(pop)
            if not folder.exists():
                continue

            image_paths.extend(sorted(folder.glob("*.jpg")))
            image_paths.extend(sorted(folder.glob("*.jpeg")))
            image_paths.extend(sorted(folder.glob("*.png")))

    return image_paths


def _prepare_metadata_dataframe(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(excel_path)

    if "FileName" not in df.columns:
        if "FilePath" in df.columns:
            df["FileName"] = df["FilePath"].astype(str).map(lambda x: Path(x).name)
        else:
            raise ValueError("Brakuje kolumny 'FileName' lub 'FilePath' w pliku metadata")

    df["FileName"] = df["FileName"].astype(str).str.strip()
    return df


def run_full_dataset_prediction(loss_name: str, model_path: str, path_manager, log_dir, full_name):
    cfg = path_manager.cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    population_mapper = PopulationMapper(cfg.data.active_populations)
    is_multitask = bool(getattr(cfg.multitask_model, "use", False))

    model = build_model(cfg).to(device)
    load_model_checkpoint(model, model_path, device)
    model.eval()

    transform = get_eval_transform(cfg)

    excel_path = path_manager.metadata_file()
    df = _prepare_metadata_dataframe(excel_path)

    data_root = path_manager.data_root()
    all_image_paths = _collect_all_image_paths(data_root, population_mapper.active_populations)

    predictions: dict[str, tuple[int | None, float | None, float | None]] = {}

    for image_path in all_image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)

                if is_multitask:
                    pop_logits, age_pred = outputs
                    probs = F.softmax(pop_logits, dim=1)[0]
                    pred_idx = int(torch.argmax(pop_logits, dim=1).item())
                    pred_class = population_mapper.to_pop(pred_idx)
                    confidence = float(probs[pred_idx].item()) * 100.0
                    predicted_age = float(age_pred.view(-1)[0].item())
                else:
                    pop_logits = outputs
                    probs = F.softmax(pop_logits, dim=1)[0]
                    pred_idx = int(torch.argmax(pop_logits, dim=1).item())
                    pred_class = population_mapper.to_pop(pred_idx)
                    confidence = float(probs[pred_idx].item()) * 100.0
                    predicted_age = None

            predictions[image_path.name.lower()] = (
                pred_class,
                round(confidence, 2),
                predicted_age,
            )

        except Exception as exc:
            print(f"Błąd przetwarzania obrazu {image_path}: {exc}")

    pred_column = f"{loss_name}_pred"
    prob_column = f"{loss_name}_prob"
    age_column = f"{loss_name}_age_pred"

    pred_classes = []
    pred_probs = []
    pred_ages = []
    not_found = []

    for file_name in df["FileName"]:
        key = str(file_name).strip().lower()
        pred = predictions.get(key, (None, None, None))
        if pred[0] is None:
            not_found.append(key)

        pred_classes.append(pred[0])
        pred_probs.append(pred[1])
        pred_ages.append(pred[2])

    new_cols = pd.DataFrame({
        pred_column: pred_classes,
        prob_column: pred_probs,
    })

    if is_multitask:
        new_cols[age_column] = pred_ages

    output_path = Path(log_dir) / f"{full_name}_predictions.xlsx"

    if output_path.exists():
        existing_df = pd.read_excel(output_path)

        for col in [pred_column, prob_column, age_column]:
            if col in existing_df.columns:
                existing_df = existing_df.drop(columns=[col], errors="ignore")

        result_df = pd.concat([existing_df, new_cols], axis=1)
    else:
        result_df = pd.concat([df, new_cols], axis=1)

    result_df.to_excel(output_path, index=False)

    print(f"\n📊 Liczba przetworzonych obrazów: {len(predictions)}")
    if not_found:
        print(f"❗Nie znaleziono predykcji dla {len(not_found)} plików metadata.")
        print(f"Przykłady brakujących plików: {not_found[:5]}")
    print(f"✅ Zapisano predykcje ({loss_name}) do: {output_path}")
    print("⏭️ Przechodzę do kolejnej funkcji straty...\n")
