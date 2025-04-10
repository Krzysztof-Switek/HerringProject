import torch.nn as nn
from torchvision import models
from omegaconf import DictConfig
import torch
import warnings


class HerringModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.cfg = config.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base = self._init_base_model().to(self.device)
        self._print_model_info()

    def _init_base_model(self) -> nn.Module:
        """Initialize base model with automatic device placement"""
        # Validate model name
        available_models = [m for m in dir(models) if m.islower() and not m.startswith('__')]
        if self.cfg.base_model not in available_models:
            raise ValueError(f"Model {self.cfg.base_model} not available. Choose from: {available_models}")

        # Handle weights
        weights = None
        if self.cfg.pretrained:
            if hasattr(models, f"{self.cfg.base_model}_Weights"):
                weights = getattr(models, f"{self.cfg.base_model}_Weights").DEFAULT
            else:
                warnings.warn(f"No weights enum for {self.cfg.base_model}, using default initialization")
                weights = "DEFAULT"

        # Initialize model
        model = getattr(models, self.cfg.base_model)(weights=weights)

        # Freeze layers if needed
        if self.cfg.freeze_encoder:
            for name, param in model.named_parameters():
                if 'fc' not in name and 'classifier' not in name:  # Don't freeze last layer
                    param.requires_grad = False

        # Replace final layer
        if hasattr(model, 'fc'):
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.cfg.num_classes)
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                num_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_features, self.cfg.num_classes)
            else:
                num_features = model.classifier.in_features
                model.classifier = nn.Linear(num_features, self.cfg.num_classes)
        else:
            raise AttributeError("Model has no recognizable classifier layer")

        return model

    def _print_model_info(self):
        """Print detailed model information"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())

        print("\n" + "=" * 50)
        print(f"Model initialized on device: {self.device}")
        print(f"Architecture: {self.cfg.base_model}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Pretrained weights: {self.cfg.pretrained}")
        print(f"Frozen encoder: {self.cfg.freeze_encoder}")
        print("=" * 50 + "\n")

    def forward(self, x):
        # Ensure input is on correct device
        if x.device != self.device:
            x = x.to(self.device)
        return self.base(x)