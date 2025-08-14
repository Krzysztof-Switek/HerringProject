import torch
import torch.nn as nn
from torchvision import models
from omegaconf import DictConfig
import warnings
from models.model_config import MODEL_CONFIGS

class MultiTaskHerringModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.full_cfg = config
        self.cfg = config.base_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base, num_features = self._init_base_model()

        clf_cfg = config.multitask_model.classifier_head
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=clf_cfg.dropout_rate),
            nn.Linear(num_features, len(self.full_cfg.data.active_populations))
        )

        reg_cfg = config.multitask_model.regression_head
        self.age_regression_head = nn.Sequential(
            nn.Linear(num_features, reg_cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=reg_cfg.dropout_rate),
            nn.Linear(reg_cfg.hidden_dim, 1)
        )

        self._print_model_info()

    def _init_base_model(self):
        model_name = self.full_cfg.model_name
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model {model_name} not supported")

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
            for pname, param in model.named_parameters():
                if not any(x in pname.lower() for x in ['fc', 'classifier', 'head']):
                    param.requires_grad = False

        classifier_path = model_details["classifier"].split('.')
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
        print(f"Architecture: {self.full_cfg.model_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Pretrained weights: {self.cfg.pretrained}")
        print(f"Frozen encoder: {self.cfg.freeze_encoder}")
        print(f"Liczba klas populacji (output): {len(self.full_cfg.data.active_populations)}")
        print("=" * 50 + "\n")
