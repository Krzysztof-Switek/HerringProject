import torch
import torch.nn as nn
from torchvision import models
from omegaconf import DictConfig
import warnings

class HerringModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.full_cfg = config
        self.cfg = config.base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base = self._init_base_model().to(self.device)
        self._print_model_info()

    def _init_base_model(self) -> nn.Module:
        model_config = {
            "resnet50": {"image_size": 224, "classifier": "fc"},
            "convnext_large": {"image_size": 384, "classifier": "classifier"},
            "vit_h_14": {"image_size": 384, "classifier": "heads"},
            "efficientnet_v2_l": {"image_size": 480, "classifier": "classifier"},
            "regnet_y_32gf": {"image_size": 384, "classifier": "fc"},
        }

        if self.cfg.base_model not in model_config:
            available = list(model_config.keys())
            raise ValueError(f"Model {self.cfg.base_model} not configured. Choose from: {available}")

        weights = None
        if self.cfg.pretrained:
            if self.cfg.base_model == "resnet50":
                weights = models.ResNet50_Weights.IMAGENET1K_V1
            elif self.cfg.base_model == "efficientnet_v2_l":
                weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
            elif self.cfg.base_model == "convnext_large":
                weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
            elif self.cfg.base_model == "vit_h_14":
                weights = models.ViT_H_14_Weights.IMAGENET1K_V1
            elif self.cfg.base_model == "regnet_y_32gf":
                weights = models.RegNet_Y_32GF_Weights.IMAGENET1K_V1
            else:
                warnings.warn(f"No weights enum for {self.cfg.base_model}, using default initialization")
                weights = "DEFAULT"

        model = getattr(models, self.cfg.base_model)(weights=weights)

        if self.cfg.freeze_encoder:
            for name, param in model.named_parameters():
                if not any(c in name.lower() for c in ['fc', 'classifier', 'head']):
                    param.requires_grad = False

        dropout_p = getattr(self.cfg, "dropout_rate", 0.0)
        classifier_path = model_config[self.cfg.base_model]["classifier"].split('.')

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
        print(f"Architecture: {self.cfg.base_model}")
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
    if cfg.multitask_model.use:
        from models.multitask_model import MultiTaskHerringModel
        return MultiTaskHerringModel(cfg)
    return HerringModel(cfg)
