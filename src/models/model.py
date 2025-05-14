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
        # Model-specific configurations
        model_config = {
            "resnet50": {"image_size": 224, "classifier": "fc"},
            "convnext_large": {"image_size": 384, "classifier": "classifier"},
            "vit_h_14": {"image_size": 384, "classifier": "heads"},
            "efficientnet_v2_l": {"image_size": 480, "classifier": "classifier"},
            "regnety_032": {"image_size": 384, "classifier": "head.fc"},
        }

        if self.cfg.base_model not in model_config:
            available_models = list(model_config.keys())
            raise ValueError(f"Model {self.cfg.base_model} not configured. Choose from: {available_models}")

        # Handle weights
        weights = None
        if self.cfg.pretrained:
            if self.cfg.base_model == "resnet50":
                # Manually load pre-trained weights for resnet50
                weights = models.ResNet50_Weights.IMAGENET1K_V1
            elif self.cfg.base_model == "efficientnet_v2_l":
                weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
            elif self.cfg.base_model == "convnext_large":
                weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
            elif self.cfg.base_model == "vit_h_14":
                weights = models.ViT_H_14_Weights.IMAGENET1K_V1
            elif self.cfg.base_model == "regnety_032":
                weights = models.RegNetY_032_Weights.IMAGENET1K_V1
            else:
                warnings.warn(f"No weights enum for {self.cfg.base_model}, using default initialization")
                weights = "DEFAULT"

        # Initialize model
        model = getattr(models, self.cfg.base_model)(weights=weights)

        # Freeze layers if needed (especially useful for transfer learning)
        if self.cfg.freeze_encoder:
            for param in model.parameters():
                param.requires_grad = False

        # Modify the classifier to match the number of classes
        dropout_p = getattr(self.cfg, "dropout_rate", 0.0)
        classifier_path = model_config[self.cfg.base_model]["classifier"].split('.')

        # Navigate to the classifier layer
        parent_module = model
        for part in classifier_path[:-1]:
            parent_module = getattr(parent_module, part)

        last_part = classifier_path[-1]
        original_classifier = getattr(parent_module, last_part)

        if isinstance(original_classifier, nn.Sequential):
            # For sequential classifiers (like in ConvNeXt)
            num_features = original_classifier[-1].in_features
            new_classifier = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(num_features, self.cfg.num_classes)
            )
            original_classifier[-1] = new_classifier
        else:
            # For simple linear classifiers
            num_features = original_classifier.in_features
            setattr(parent_module, last_part, nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(num_features, self.cfg.num_classes)
            ))

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
        print(f"Dropout rate: {getattr(self.cfg, 'dropout_rate', 0.0)}")
        print("=" * 50 + "\n")

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)
        return self.base(x)
