import cv2
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
from models.model import HerringModel
from models.grad_cam import GradCAM
from omegaconf import DictConfig, OmegaConf


class Predictor:
    def __init__(self, config_path: str = None):
        """
        Inicjalizacja predykatora z obsługą Grad-CAM
        Args:
            config_path: Ścieżka do pliku konfiguracyjnego
        """
        # 1. Ładowanie konfiguracji
        self.project_root = Path(__file__).parent.parent
        self.cfg = self._load_config(config_path)

        # 2. Inicjalizacja urządzenia
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 3. Ładowanie modelu
        self.model = self._load_model()
        self.model.eval()

        # 4. Przygotowanie transformacji
        self.transform = transforms.Compose([
            transforms.Resize(self.cfg['data']['image_size']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def _load_config(self, config_path):
        """Ładowanie konfiguracji z walidacją ścieżek"""
        if config_path is None:
            config_path = self.project_root / "src" / "config" / "config.yaml"

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        cfg = OmegaConf.load(config_path)
        cfg.data.root_dir = str(self.project_root / cfg.data.root_dir)
        cfg.training.checkpoint_dir = str(self.project_root / cfg.training.checkpoint_dir)
        cfg.prediction.model_path = str(self.project_root / cfg.prediction.model_path)
        cfg.prediction.image_path = str(self.project_root / cfg.prediction.image_path)
        cfg.prediction.results_dir = str(self.project_root / cfg.prediction.results_dir)

        return cfg

    def _load_model(self):
        """Ładowanie modelu z checkpointu"""
        model_path = Path(self.cfg.prediction.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")

        model = HerringModel(DictConfig(self.cfg)).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        return model

    def predict_with_gradcam(self, image_path: str = None):
        """
        Wykonanie predykcji z wizualizacją Grad-CAM
        Args:
            image_path: Opcjonalna ścieżka do obrazu (domyślnie z configa)
        """
        # 1. Przygotowanie ścieżek
        image_path = Path(image_path or self.cfg['prediction']['image_path'])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        results_dir = Path(self.cfg['prediction']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)

        # 2. Wczytanie i transformacja obrazu
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 3. Predykcja klasy
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
            pred_class = output.argmax().item()

        print(f"\nPrediction results:")
        print(f"- Predicted class: {pred_class}")
        print(f"- Confidence: {probabilities[pred_class]:.1f}%")

        # 4. Generowanie Grad-CAM
        grad_cam = GradCAM(self.model, self.cfg['gradcam']['target_layer'])
        heatmap = grad_cam.generate(input_tensor, pred_class)

        # 5. Wizualizacja i zapis wyników
        self._visualize_results(image, heatmap, pred_class, probabilities[pred_class], results_dir)

    def _visualize_results(self, image, heatmap, pred_class, confidence, results_dir):
        """Wizualizacja i zapis wyników"""
        # 1. Przygotowanie danych
        original_img = np.array(image)
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

        # 2. Tworzenie heatmapy
        colormap = plt.get_cmap(self.cfg['gradcam']['colormap'])
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = colormap(heatmap)[:, :, :3] * 255

        # 3. Nałożenie heatmapy
        superimposed_img = (original_img * (1 - self.cfg['gradcam']['alpha']) +
                            heatmap * self.cfg['gradcam']['alpha'])
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        # 4. Wyświetlenie
        if self.cfg['prediction']['show_visualization']:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(original_img)
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(heatmap)
            plt.title("Grad-CAM Heatmap")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(superimposed_img)
            plt.title(f"Predicted: {pred_class}\nConfidence: {confidence:.1f}%")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        # 5. Zapis wyników
        if self.cfg['prediction']['save_results']:
            output_path = results_dir / f"gradcam_{Path(self.cfg['prediction']['image_path']).stem}.png"
            plt.imsave(output_path, superimposed_img)
            print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    try:
        print("==== Herring Otolith Grad-CAM Visualization ====")
        predictor = Predictor()
        predictor.predict_with_gradcam()
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        raise