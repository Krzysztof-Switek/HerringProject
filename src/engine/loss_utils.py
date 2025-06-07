import torch
import torch.nn as nn

class LossFactory:
    def __init__(self, loss_type: str = "sample_weighted_ce"):
        self.loss_type = loss_type

    def get(self):
        if self.loss_type == "sample_weighted_ce":
            return SampleWeightedCrossEntropy()
        else:
            raise ValueError(f"Nieznany typ funkcji straty: {self.loss_type}")

class SampleWeightedCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, meta=None):
        loss = nn.functional.cross_entropy(outputs, targets, reduction='none')

        # Jeśli meta nie jest podane, nie ważymy próbek
        if meta is None:
            return loss.mean()

        weight_map = {
            (2, 3): 1.3,
            (2, 4): 1.5,
            (2, 5): 1.6,
            (2, 6): 1.8,
            (2, 7): 2.0,
            (2, 8): 2.0,
            (2, 9): 2.0,
            (2, 10): 2.0
        }
        weights = [
            weight_map.get((int(p), int(w)), 1.0)
            for p, w in zip(meta['populacja'], meta['wiek'])
        ]
        weights = torch.tensor(weights, dtype=torch.float32).to(outputs.device)
        return (loss * weights).mean()
