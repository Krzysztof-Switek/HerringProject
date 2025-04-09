import torch
from omegaconf import OmegaConf
from torch import nn
from data_loader.dataset import HerringDataset
from models.model import HerringModel


class Trainer:
    def __init__(self, config_path="config/config.yaml"):
        self.cfg = OmegaConf.load(config_path)
        self.device = torch.device(self.cfg.training.device)
        self.model = HerringModel(self.cfg).to(self.device)
        self.dataset = HerringDataset(self.cfg)

    def train(self):
        train_loader, val_loader, classes = self.dataset.get_loaders()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.cfg.training.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.cfg.training.epochs):
            self.model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{self.cfg.training.epochs} completed")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()