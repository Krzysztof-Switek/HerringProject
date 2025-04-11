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
import os
from typing import List, Tuple, Optional


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

        # 5. Inicjalizacja zmiennych dla nawigacji
        self.current_image_index = 0
        self.image_paths = []
        self._init_image_paths()

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

    def _init_image_paths(self):
        """Inicjalizacja listy ścieżek do obrazów"""
        image_dir = self.cfg['prediction_modes']['navigation']['image_dir']
        if not Path(image_dir).is_absolute():
            image_dir = self.project_root / image_dir

        self.image_paths = self._get_image_paths_from_dir(image_dir)

        # Ustaw aktualny indeks na podstawie domyślnego obrazu z configa
        default_image = Path(self.cfg.prediction.image_path)
        if not default_image.is_absolute():
            default_image = self.project_root / default_image

        try:
            self.current_image_index = self.image_paths.index(str(default_image))
        except ValueError:
            self.current_image_index = 0

    def _get_image_paths_from_dir(self, dir_path: str) -> List[str]:
        """Pobiera posortowaną listę ścieżek do obrazów w katalogu"""
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = sorted([str(p.resolve()) for p in dir_path.glob('*')
                              if p.suffix.lower() in extensions])
        return image_paths

    def predict_with_gradcam(self, image_path: str = None, mode: str = None):
        """
        Wykonanie predykcji z wizualizacją Grad-CAM
        Args:
            image_path: Opcjonalna ścieżka do obrazu (domyślnie z configa)
            mode: Tryb pracy ('single' lub 'batch')
        """
        mode = mode or self.cfg['prediction_modes']['mode']

        if mode == "batch":
            self._batch_mode()
        else:
            if image_path is None and self.image_paths:
                image_path = self.image_paths[self.current_image_index]
            self._single_mode(image_path)

    def _single_mode(self, image_path: str):
        """Tryb pojedynczego obrazu z nawigacją"""
        # 1. Przygotowanie ścieżek
        image_path = Path(image_path)
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

        print(f"\nPrediction results for {image_path.name}:")
        print(f"- Predicted class: {pred_class}")
        print(f"- Confidence: {probabilities[pred_class]:.1f}%")

        # 4. Generowanie Grad-CAM
        grad_cam = GradCAM(self.model, self.cfg['gradcam']['target_layer'])
        heatmap = grad_cam.generate(input_tensor, pred_class)
        grad_cam.clear_hooks()

        # 5. Wizualizacja i zapis wyników
        self._visualize_single(image, heatmap, pred_class, probabilities[pred_class],
                               results_dir, image_path.name)

        # 6. Nawigacja jeśli jest więcej obrazów
        if len(self.image_paths) > 1:
            self._handle_navigation()

    def _batch_mode(self):
        """Tryb wsadowy z wieloma obrazami spełniającymi warunek pewności"""
        min_confidence = self.cfg['prediction_modes']['batch']['min_confidence']
        num_images = self.cfg['prediction_modes']['batch']['num_images']

        print(f"\nStarting batch mode processing (min confidence: {min_confidence}%)...")

        # 1. Przygotowanie wyników
        results = []
        processed = 0

        # 2. Przetwarzanie obrazów (ograniczone do pierwszych 500 dla wydajności)
        for img_path in self.image_paths[:500]:
            try:
                pred_class, confidence = self._get_prediction_info(img_path)
                if confidence >= min_confidence:
                    results.append((img_path, pred_class, confidence))

                processed += 1
                if processed % 50 == 0:
                    print(f"Processed {processed} images, found {len(results)} with confidence >= {min_confidence}%")

                if len(results) >= num_images * 2:  # Zbieramy 2x więcej niż potrzebujemy dla lepszej selekcji
                    break

            except Exception as e:
                print(f"Error processing {Path(img_path).name}: {str(e)}")
                continue

        # 3. Sortowanie i wybór najlepszych obrazów
        results.sort(key=lambda x: x[2], reverse=True)
        selected_results = results[:num_images]

        if not selected_results:
            print(f"\nNo images found with confidence >= {min_confidence}%")
            return

        print(f"\nFound {len(selected_results)} images meeting criteria:")
        for img_path, pred_class, confidence in selected_results:
            print(f"- {Path(img_path).name}: class {pred_class}, confidence {confidence:.1f}%")

        # 4. Generowanie wizualizacji
        self._visualize_batch(selected_results)

    def _get_prediction_info(self, image_path: str) -> Tuple[int, float]:
        """Pobiera informacje o predykcji dla obrazu (bez Grad-CAM)"""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
            pred_class = output.argmax().item()

        return pred_class, float(probabilities[pred_class])

    def _visualize_single(self, image, heatmap, pred_class, confidence,
                          results_dir: Path, image_name: str):
        """Wizualizacja pojedynczego obrazu z Grad-CAM"""
        # 1. Przygotowanie danych
        original_img = np.array(image)
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

        # 2. Tworzenie heatmapy
        colormap = plt.get_cmap(self.cfg['gradcam']['colormap'])
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = colormap(heatmap)[:, :, :3] * 255

        # 3. Nałożenie heatmapy na oryginalny obraz
        superimposed_img = (original_img * (1 - self.cfg['gradcam']['alpha']) +
                            heatmap * self.cfg['gradcam']['alpha'])
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.float32) / 255.0

        # 4. Wyświetlenie
        if self.cfg['prediction']['show_visualization']:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(original_img / 255.0)
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(heatmap / 255.0)
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
            output_path = results_dir / f"gradcam_{Path(image_name).stem}.png"
            plt.imsave(output_path, superimposed_img)
            print(f"Results saved to: {output_path}")

    def _visualize_batch(self, images_info: List[Tuple[str, int, float]]):
        """Wizualizacja wsadowa wielu obrazów"""
        plt.figure(figsize=(15, 10))
        num_images = len(images_info)
        cols = min(4, num_images)
        rows = (num_images + cols - 1) // cols

        for i, (img_path, pred_class, confidence) in enumerate(images_info, 1):
            try:
                # 1. Wczytanie i przetworzenie obrazu
                image = Image.open(img_path).convert('RGB')
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)

                # 2. Generowanie Grad-CAM
                grad_cam = GradCAM(self.model, self.cfg['gradcam']['target_layer'])
                heatmap = grad_cam.generate(input_tensor, pred_class)
                grad_cam.clear_hooks()

                # 3. Przygotowanie wizualizacji
                original_img = np.array(image)
                heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
                colormap = plt.get_cmap(self.cfg['gradcam']['colormap'])
                heatmap = (heatmap * 255).astype(np.uint8)
                heatmap = colormap(heatmap)[:, :, :3] * 255
                superimposed_img = (original_img * (1 - self.cfg['gradcam']['alpha']) +
                                    heatmap * self.cfg['gradcam']['alpha'])
                superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.float32) / 255.0

                # 4. Dodanie do wykresu
                plt.subplot(rows, cols, i)
                plt.imshow(superimposed_img)
                plt.title(f"{Path(img_path).name}\nClass: {pred_class} ({confidence:.1f}%)")
                plt.axis('off')

            except Exception as e:
                print(f"Error visualizing {Path(img_path).name}: {str(e)}")
                continue

        plt.tight_layout()

        # Zapis wyników
        if self.cfg['prediction']['save_results']:
            results_dir = Path(self.cfg['prediction']['results_dir'])
            results_dir.mkdir(parents=True, exist_ok=True)

            output_path = results_dir / "batch_gradcam_results.png"
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"\nBatch results saved to: {output_path}")

        if self.cfg['prediction']['show_visualization']:
            plt.show()

    def _handle_navigation(self):
        """Obsługa nawigacji między obrazami w trybie single"""
        print("\nNavigation options:")
        print("- [N] Next image")
        print("- [P] Previous image")
        print("- [Q] Quit")

        while True:
            key = input("Enter your choice (N/P/Q): ").strip().upper()

            if key == 'Q':
                break
            elif key == 'N':
                self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
            elif key == 'P':
                self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
            else:
                print("Invalid choice. Please try again.")
                continue

            next_image = self.image_paths[self.current_image_index]
            self._single_mode(next_image)
            break


if __name__ == "__main__":
    try:
        print("==== Herring Otolith Grad-CAM Visualization ====")
        predictor = Predictor()

        # Wybór trybu pracy
        mode = input("Choose mode [single/batch] (default: single): ").strip().lower()
        if mode not in ['single', 'batch']:
            mode = 'single'

        predictor.predict_with_gradcam(mode=mode)

    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        raise