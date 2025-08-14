import torch
import torch.nn as nn
from torchvision import models
from omegaconf import DictConfig
import warnings
from models.model_config import MODEL_CONFIGS


class HerringModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.full_cfg = config
        self.cfg = config.base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base = self._init_base_model().to(self.device)
        self._print_model_info()

    def _init_base_model(self) -> nn.Module:
        model_name = self.full_cfg.model_name
        if model_name not in MODEL_CONFIGS:
            available = list(MODEL_CONFIGS.keys())
            raise ValueError(f"Model {model_name} not configured. Choose from: {available}")

        model_details = MODEL_CONFIGS[model_name]

        weights = None
        if self.cfg.pretrained:
            try:
                weights = getattr(models, model_details["weights"])
            except AttributeError:
                warnings.warn(f"No weights enum for {model_name}, using default initialization")
                weights = "DEFAULT"

        model = getattr(models, model_name)(weights=weights)

        if self.cfg.freeze_encoder:
            for name, param in model.named_parameters():
                if not any(c in name.lower() for c in ['fc', 'classifier', 'head']):
                    param.requires_grad = False

        dropout_p = getattr(self.cfg, "dropout_rate", 0.0)
        classifier_path = model_details["classifier"].split('.')

        parent = model
        for part in classifier_path[:-1]:
            parent = getattr(parent, part)

        last = classifier_path[-1]
        original = getattr(parent, last)

        if isinstance(original, nn.Sequential):
            num_features = original[-1].in_features
            new_classifier = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(num_features, len(self.full_cfg.data.active_populations))
            )
            original[-1] = new_classifier
        else:
            num_features = original.in_features
            setattr(parent, last, nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(num_features, len(self.full_cfg.data.active_populations))
            ))

        return model

    def _print_model_info(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print("\n" + "=" * 50)
        print(f"Model initialized on device: {self.device}")
        print(f"Architecture: {self.full_cfg.model_name}")
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Pretrained weights: {self.cfg.pretrained}")
        print(f"Frozen encoder: {self.cfg.freeze_encoder}")
        print(f"Dropout rate: {getattr(self.cfg, 'dropout_rate', 0.0)}")
        print("=" * 50 + "\n")

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)
        return self.base(x)


def build_model(cfg: DictConfig) -> nn.Module:
    """Builds the model based on the mode specified in the config."""
    if cfg.mode == "multitask":
        from models.multitask_model import MultiTaskHerringModel
        return MultiTaskHerringModel(cfg)
    elif cfg.mode == "single":
        return HerringModel(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")
