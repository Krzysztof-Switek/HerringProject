import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.model import HerringModel
from src.models.model_heatmaps import GradCAM, GradCAMPP, GuidedBackprop
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import cv2

def generate_heatmaps(category: str, rows, model, transform, methods, cfg, model_name, device):
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "results" / model_name / category
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in rows.iterrows():
        img_path = Path(row["FilePath"])
        if not img_path.exists():
            print(f"❌ Error: {img_path} not found.")
            continue

        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        try:
            true_class = int(row["Populacja"])
            pred_class_excel = int(row[f"{model_name}_pred"])
            if true_class not in [1, 2] or pred_class_excel not in [1, 2]:
                print(f"⚠️ Invalid class for {img_path.name}")
                continue
            pred_class = 0 if pred_class_excel == 1 else 1
        except Exception as e:
            print(f"⚠️ Error parsing class for {img_path.name}: {e}")
            continue

        confidence = float(row[f"{model_name}_probability"])
        file_name = img_path.name

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(np.array(image))
        axes[0].set_title("Analyzed picture")
        axes[0].axis('off')

        for i, (name, method) in enumerate(methods.items(), 1):
            try:
                heatmap = method.generate(input_tensor, pred_class)
                if name == "guided_backprop":
                    axes[i].imshow(heatmap, cmap='gray')
                else:
                    heatmap = cv2.resize(heatmap, image.size)
                    axes[i].imshow(image)
                    axes[i].imshow(heatmap, cmap=cfg.visualization.colormap, alpha=cfg.visualization.alpha)
                axes[i].set_title(name.upper())
                axes[i].axis('off')
            except Exception as e:
                print(f"⚠️ {name} failed: {e}")
                axes[i].imshow(np.zeros_like(image))
                axes[i].set_title(f"{name.upper()} (error)")
                axes[i].axis('off')

        fig.suptitle(
            f"Model: {model_name.upper()} | Pred: {pred_class_excel} | True: {true_class} | Prob: {confidence:.1f}%",
            fontsize=16, fontweight="bold", y=1.05
        )
        fig.subplots_adjust(top=0.8)

        output_path = output_dir / f"{file_name.replace('.jpg', '')}_heatmap.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"✅ Saved: {category}/{output_path.name}")

def generate_all_heatmaps():
    project_root = Path(__file__).parent.parent
    config_path = project_root / "src" / "config" / "config.yaml"
    cfg = OmegaConf.load(config_path)

    model_name = cfg.model.base_model.replace("/", "_").lower()
    model_path = project_root / cfg.prediction.model_path
    excel_path = project_root / "src" / "data_loader" / f"AnalysisWithOtolithPhoto_with_sets_with_preds_{model_name}.xlsx"

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HerringModel(cfg).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    # Heatmapy
    methods = {
        "gradcam": GradCAM(model, cfg.visualization.target_layer),
        "gradcam++": GradCAMPP(model, cfg.visualization.target_layer),
        "guided_backprop": GuidedBackprop(model)
    }

    transform = transforms.Compose([
        transforms.Resize(cfg.data.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    df = pd.read_excel(excel_path)
    pred_col = f"{model_name}_pred"
    prob_col = f"{model_name}_probability"

    df = df[df[pred_col].notnull() & df[prob_col].notnull()]
    correct = df[df["Populacja"] == df[pred_col]]
    wrong = df[df["Populacja"] != df[pred_col]]

    top10 = correct.sort_values(by=prob_col, ascending=False).head(10)
    bottom10 = correct.sort_values(by=prob_col, ascending=True).head(10)
    wrong10 = wrong.sort_values(by=prob_col, ascending=False).head(10)

    generate_heatmaps("top_10", top10, model, transform, methods, cfg, model_name, device)
    generate_heatmaps("bottom_10", bottom10, model, transform, methods, cfg, model_name, device)
    generate_heatmaps("wrong_10", wrong10, model, transform, methods, cfg, model_name, device)

if __name__ == "__main__":
    generate_all_heatmaps()
