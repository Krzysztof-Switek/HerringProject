import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
from models.model import HerringModel
from src.models.model_heatmaps import GradCAM, GradCAMPP, GuidedBackprop
from omegaconf import DictConfig, OmegaConf
from typing import List, Tuple, Optional


class Predictor:
    def __init__(self, config_path: str = None):
        self.project_root = Path(__file__).parent.parent
        self.cfg = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(self.cfg['data']['image_size']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.visualization_methods = {
            'gradcam': GradCAM,
            'gradcam++': GradCAMPP,
            'guided_backprop': GuidedBackprop
        }

        self.current_image_index = 0
        self.image_paths = []
        self._init_image_paths()

    def _load_config(self, config_path):
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

        if 'visualization' not in cfg:
            cfg.visualization = OmegaConf.create({
                'methods': ['gradcam', 'gradcam++', 'guided_backprop'],
                'target_layer': 'base.layer4.2.conv3',
                'colormap': 'jet',
                'alpha': 0.5,
                'matrix_cols': 3,
                'save_individual': False
            })

        return cfg

    def _load_model(self):
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
        image_dir = self.cfg['prediction_modes']['navigation']['image_dir']
        if not Path(image_dir).is_absolute():
            image_dir = self.project_root / image_dir

        self.image_paths = self._get_image_paths_from_dir(image_dir)

        default_image = Path(self.cfg.prediction.image_path)
        if not default_image.is_absolute():
            default_image = self.project_root / default_image

        try:
            self.current_image_index = self.image_paths.index(str(default_image))
        except ValueError:
            self.current_image_index = 0

    def _get_image_paths_from_dir(self, dir_path: str) -> List[str]:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        return sorted([str(p.resolve()) for p in dir_path.glob('*')
                       if p.suffix.lower() in extensions])

    def predict_with_visualization(self, image_path: str = None, mode: str = None):
        mode = mode or self.cfg['prediction_modes']['mode']
        if mode == "batch":
            self._batch_mode()
        else:
            if image_path is None and self.image_paths:
                image_path = self.image_paths[self.current_image_index]
            self._single_mode(image_path)

    def _single_mode(self, image_path: str):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        results_dir = Path(self.cfg['prediction']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
            pred_class = output.argmax().item()

        print(f"\n[PREDICTION] {image_path.name}:")
        print(f"- Class: {pred_class}")
        print(f"- Confidence: {probabilities[pred_class]:.1f}%")

        self._visualize_matrix(
            image=image,
            input_tensor=input_tensor,
            pred_class=pred_class,
            confidence=probabilities[pred_class],
            results_dir=results_dir,
            image_name=image_path.name
        )

        if len(self.image_paths) > 1:
            self._handle_navigation()

    def _visualize_matrix(self, image, input_tensor, pred_class, confidence,
                          results_dir: Path, image_name: str):
        original_img = np.array(image)
        methods = self.cfg['visualization']['methods']
        model_name = self.cfg['model']['base_model']

        # Calculate figure size based on number of methods
        cols = len(methods) + 1
        fig_width = 5 * cols
        fig, axes = plt.subplots(1, cols, figsize=(fig_width, 5))

        # Window management
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state('zoomed')
        except:
            try:
                mng.window.showMaximized()
            except:
                pass

        axes = axes.flatten()

        # Original image
        axes[0].imshow(original_img / 255.0)
        axes[0].set_title("Analyzed picture", fontsize=12, pad=10, y=-0.15)
        axes[0].axis('off')

        # Visualization methods
        for i, method_name in enumerate(methods, 1):
            try:
                visualizer = (self.visualization_methods[method_name](self.model)
                              if method_name == 'guided_backprop'
                              else self.visualization_methods[method_name](
                    self.model, self.cfg['visualization']['target_layer']))

                heatmap = visualizer.generate(input_tensor, pred_class)

                if method_name == 'guided_backprop':
                    axes[i].imshow(heatmap, cmap='gray')
                else:
                    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
                    axes[i].imshow(original_img / 255.0)
                    axes[i].imshow(heatmap,
                                   cmap=self.cfg['visualization']['colormap'],
                                   alpha=self.cfg['visualization']['alpha'])

                axes[i].set_title(method_name.upper(), fontsize=12, pad=10, y=-0.15)
                axes[i].axis('off')

            except Exception as e:
                axes[i].imshow(np.zeros_like(original_img))
                axes[i].set_title(f"{method_name.upper()} (failed)", fontsize=12, pad=10, y=-0.15)
                axes[i].axis('off')
                continue

        # Main title and subtitle
        main_title = f"Model: {model_name.upper()} | Predicted class: {pred_class} | Accuracy: {confidence:.1f}%"
        filename = f"File name: {Path(image_name).name}"

        # Adjust layout parameters - manual layout instead of tight_layout
        plt.subplots_adjust(
            top=0.85,  # Top margin
            bottom=0.05,  # Bottom margin
            left=0.05 / cols,  # Dynamic left margin
            right=1 - 0.05 / cols,  # Dynamic right margin
            wspace=0.3 + 0.1 * cols  # Dynamic spacing based on column count
        )

        # Main title
        fig.suptitle(
            main_title,
            fontsize=14,
            fontweight='bold',
            y=0.95,  # Slightly higher position
            x=0.5,
            ha='center'
        )

        # Subtitle
        plt.text(
            0.5, 0.90,
            filename,
            fontsize=12,
            ha='center',
            va='center',
            transform=fig.transFigure
        )

        # Save and show
        if self.cfg['prediction']['save_results']:
            output_path = results_dir / f"{model_name}_{Path(image_name).stem}.png"
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"\n[SAVED] Results to: {output_path}")

        if self.cfg['prediction']['show_visualization']:
            plt.show()

    def _batch_mode(self):
        min_confidence = self.cfg['prediction_modes']['batch']['min_confidence']
        num_images = self.cfg['prediction_modes']['batch']['num_images']

        print(f"\n[BATCH MODE] Processing (min confidence: {min_confidence}%)...")

        results = []
        processed = 0

        for img_path in self.image_paths[:500]:
            try:
                pred_class, confidence = self._get_prediction_info(img_path)
                if confidence >= min_confidence:
                    results.append((img_path, pred_class, confidence))

                processed += 1
                if processed % 50 == 0:
                    print(f"Processed {processed} images, found {len(results)} matches")

                if len(results) >= num_images * 2:
                    break

            except Exception as e:
                print(f"Error processing {Path(img_path).name}: {str(e)}")
                continue

        results.sort(key=lambda x: x[2], reverse=True)
        selected_results = results[:num_images]

        if not selected_results:
            print(f"\nNo images found with confidence >= {min_confidence}%")
            return

        print(f"\nFound {len(selected_results)} images meeting criteria:")
        for img_path, pred_class, confidence in selected_results:
            print(f"- {Path(img_path).name}: class {pred_class}, confidence {confidence:.1f}%")

        self._visualize_batch(selected_results)

    def _get_prediction_info(self, image_path: str) -> Tuple[int, float]:
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
            pred_class = output.argmax().item()

        return pred_class, float(probabilities[pred_class])

    def _visualize_batch(self, images_info: List[Tuple[str, int, float]]):
        plt.figure(figsize=(15, 10))
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state('zoomed')
        except Exception:
            try:
                mng.window.showMaximized()
            except Exception:
                pass

        num_images = len(images_info)
        cols = min(4, num_images)
        rows = (num_images + cols - 1) // cols

        for i, (img_path, pred_class, confidence) in enumerate(images_info, 1):
            try:
                image = Image.open(img_path).convert('RGB')
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)

                grad_cam = GradCAM(self.model, self.cfg['visualization']['target_layer'])
                print(f"\n[HOOK STATUS] {grad_cam.get_hook_status()}")
                heatmap = grad_cam.generate(input_tensor, pred_class)
                grad_cam.clear_hooks()

                original_img = np.array(image)
                heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
                colormap = plt.get_cmap(self.cfg['visualization']['colormap'])
                heatmap = (heatmap * 255).astype(np.uint8)
                heatmap = colormap(heatmap)[:, :, :3] * 255
                superimposed_img = (original_img * (1 - self.cfg['visualization']['alpha']) +
                                    heatmap * self.cfg['visualization']['alpha'])
                superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.float32) / 255.0

                plt.subplot(rows, cols, i)
                plt.imshow(superimposed_img)
                plt.title(f"{Path(img_path).name}\nClass: {pred_class} ({confidence:.1f}%)",
                          fontsize=10, pad=10, y=-0.2)  # y ujemne przesuwa podpis pod obrazek
                plt.axis('off')

            except Exception as e:
                print(f"Error visualizing {Path(img_path).name}: {str(e)}")
                continue

        plt.tight_layout(h_pad=2.0)  # Zwiększony odstęp pionowy
        plt.subplots_adjust(top=0.9, bottom=0.1)  # Dostosowane marginesy

        if self.cfg['prediction']['save_results']:
            results_dir = Path(self.cfg['prediction']['results_dir'])
            results_dir.mkdir(parents=True, exist_ok=True)

            output_path = results_dir / "results-change.png"
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"\n[SAVED] Batch results to: {output_path}")

        if self.cfg['prediction']['show_visualization']:
            plt.show()

    def _handle_navigation(self):
        print("\n[NAVIGATION] Options:")
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
        print("==== Herring Otolith Visualization Matrix ====")
        predictor = Predictor()

        mode = input("Choose mode [single/batch] (default: single): ").strip().lower()
        if mode not in ['single', 'batch']:
            mode = 'single'

        predictor.predict_with_visualization(mode=mode)

    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}")
        raise
