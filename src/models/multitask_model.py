import torch
import torch.nn as nn
from torchvision import models
from omegaconf import DictConfig
import warnings

class MultiTaskHerringModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.full_cfg = config
        self.cfg = config.multitask_model.backbone_model  # 🔧 Nowa ścieżka

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base, num_features = self._init_base_model()

        # Główny klasyfikator populacji
        clf_cfg = config.multitask_model.classifier_head  # 🔧
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=clf_cfg.dropout_rate),
            nn.Linear(num_features, len(self.full_cfg.data.active_populations))
        )

        # Głowa regresji wieku
        reg_cfg = config.multitask_model.regression_head  # 🔧
        self.age_regression_head = nn.Sequential(
            nn.Linear(num_features, reg_cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=reg_cfg.dropout_rate),
            nn.Linear(reg_cfg.hidden_dim, 1)
        )

        self._print_model_info()

    def _init_base_model(self):
        model_config = {
            "resnet50": {"classifier": "fc"},
            "convnext_large": {"classifier": "classifier"},
            #"vit_h_14": {"classifier": "heads"},
            "efficientnet_v2_l": {"classifier": "classifier"},
            "regnet_y_32gf": {"classifier": "fc"},
        }

        name = self.cfg.model_name  # 🔧
        if name not in model_config:
            raise ValueError(f"Model {name} not supported")

        weights = None
        if self.cfg.pretrained:
            weight_map = {
                "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
                "efficientnet_v2_l": models.EfficientNet_V2_L_Weights.IMAGENET1K_V1,
                "convnext_large": models.ConvNeXt_Large_Weights.IMAGENET1K_V1,
                #"vit_h_14": models.ViT_H_14_Weights.IMAGENET1K_V1,
                "regnet_y_32gf": models.RegNet_Y_32GF_Weights.IMAGENET1K_V1
            }
            weights = weight_map.get(name, None)

        model = getattr(models, name)(weights=weights)

        if self.cfg.freeze_encoder:
            for pname, param in model.named_parameters():
                if not any(x in pname.lower() for x in ['fc', 'classifier', 'head']):
                    param.requires_grad = False

        classifier_path = model_config[name]["classifier"].split('.')
        parent = model
        for part in classifier_path[:-1]:
            parent = getattr(parent, part)
        last = classifier_path[-1]
        classifier = getattr(parent, last)

        num_features = classifier[-1].in_features if isinstance(classifier, nn.Sequential) else classifier.in_features
        setattr(parent, last, nn.Identity())

        return model, num_features

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)
        features = self.base(x)
        if isinstance(features, (tuple, list)):
            features = features[0]

        logits = self.classifier_head(features)
        age_pred = self.age_regression_head(features).squeeze(1)
        return logits, age_pred

    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\n" + "=" * 50)
        print(f"[MultiTask Model] initialized on device: {self.device}")
        print(f"Architecture: {self.cfg.model_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Pretrained weights: {self.cfg.pretrained}")
        print(f"Frozen encoder: {self.cfg.freeze_encoder}")
        print(f"Liczba klas populacji (output): {len(self.full_cfg.data.active_populations)}")
        print("=" * 50 + "\n")
