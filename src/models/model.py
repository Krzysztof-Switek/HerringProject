import torch.nn as nn
from torchvision import models
from omegaconf import DictConfig


class HerringModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.cfg = config.model
        self.base = self._init_base_model()

    def _init_base_model(self):
        model = getattr(models, self.cfg.base_model)(
            pretrained=self.cfg.pretrained)

        if self.cfg.freeze_encoder:
            for param in model.parameters():
                param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.cfg.num_classes)
        return model

    def forward(self, x):
        return self.base(x)