import torch
import torch.nn as nn
from omegaconf import DictConfig

try:
    import timm
except ImportError:
    raise ImportError("timm nie jest zainstalowane. Uruchom: pip install timm>=0.9.16")


TIMM_MODELS = {
    "maxvit_base_tf_512",
    "vit_large_patch14_dinov2",
}

_TIMM_FULL_IDS = {
    "maxvit_base_tf_512": "maxvit_base_tf_512.in21k_ft_in1k",
    "vit_large_patch14_dinov2": "vit_large_patch14_dinov2.lvd142m",
}


def _create_timm_backbone(model_name: str, image_size: int, pretrained: bool):
    full_id = _TIMM_FULL_IDS.get(model_name, model_name)
    model = timm.create_model(
        full_id,
        pretrained=pretrained,
        num_classes=0,
        img_size=image_size,
    )
    return model, model.num_features


class TimmHerringModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.full_cfg = config
        self.cfg = config.base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base, num_features = _create_timm_backbone(
            model_name=self.cfg.base_model,
            image_size=self.cfg.image_size,
            pretrained=self.cfg.pretrained,
        )

        if self.cfg.freeze_encoder:
            for param in self.base.parameters():
                param.requires_grad = False

        dropout_p = getattr(self.cfg, "dropout_rate", 0.0)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_features, len(self.full_cfg.data.active_populations)),
        )

        self._print_model_info()

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)
        return self.classifier(self.base(x))

    def _print_model_info(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\n" + "=" * 50)
        print(f"[TimmHerringModel] initialized on device: {self.device}")
        print(f"Architecture: {self.cfg.base_model}")
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Pretrained weights: {self.cfg.pretrained}")
        print(f"Frozen encoder: {self.cfg.freeze_encoder}")
        print("=" * 50 + "\n")


class TimmMultiTaskHerringModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.full_cfg = config
        self.cfg = config.multitask_model.backbone_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base, num_features = _create_timm_backbone(
            model_name=self.cfg.model_name,
            image_size=self.cfg.image_size,
            pretrained=self.cfg.pretrained,
        )

        if self.cfg.freeze_encoder:
            for param in self.base.parameters():
                param.requires_grad = False

        clf_cfg = config.multitask_model.classifier_head
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=clf_cfg.dropout_rate),
            nn.Linear(num_features, len(self.full_cfg.data.active_populations)),
        )

        reg_cfg = config.multitask_model.regression_head
        self.age_regression_head = nn.Sequential(
            nn.Linear(num_features, reg_cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=reg_cfg.dropout_rate),
            nn.Linear(reg_cfg.hidden_dim, 1),
        )

        self._print_model_info()

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)
        features = self.base(x)
        logits = self.classifier_head(features)
        age_pred = self.age_regression_head(features).squeeze(1)
        return logits, age_pred

    def _print_model_info(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\n" + "=" * 50)
        print(f"[TimmMultiTaskHerringModel] initialized on device: {self.device}")
        print(f"Architecture: {self.cfg.model_name}")
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Pretrained weights: {self.cfg.pretrained}")
        print(f"Frozen encoder: {self.cfg.freeze_encoder}")
        print(f"Liczba klas populacji (output): {len(self.full_cfg.data.active_populations)}")
        print("=" * 50 + "\n")