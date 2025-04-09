import torch
from PIL import Image
from mpmath.identification import transforms
from omegaconf import OmegaConf
from models.model import HerringModel
from models.grad_cam import GradCAM
from utils.visualize import Visualizer


class Predictor:
    def __init__(self, config_path="config/config.yaml", model_path=None):
        self.cfg = OmegaConf.load(config_path)
        self.device = torch.device(self.cfg.training.device)
        self.model = HerringModel(self.cfg).to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, image_path):
        transform = transforms.Compose([
            transforms.Resize(self.cfg.data.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            pred_class = output.argmax().item()

        target_layer = self.model.base.layer4[-1].conv3
        cam = GradCAM(self.model, target_layer)
        heatmap = cam.generate(input_tensor)

        Visualizer.plot_gradcam(input_tensor.cpu().squeeze(), heatmap, str(pred_class))


if __name__ == "__main__":
    predictor = Predictor(model_path="../../checkpoints/model.pth")
    predictor.predict("../../data/test/sample.jpg")