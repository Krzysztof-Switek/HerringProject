import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


class LossFactory:
    def __init__(self, loss_type: str, **kwargs):
        self.loss_type = loss_type
        self.kwargs = kwargs

    def get(self):
        if self.loss_type == "standard_ce":
            return StandardCrossEntropy()
        elif self.loss_type == "sample_weighted_ce":
            return SampleWeightedCrossEntropy()
        elif self.loss_type == "weighted_age_ce":
            return WeightedAgeCrossEntropy()
        elif self.loss_type == "focal_loss_ageboost":
            return FocalLossWithAgeBoost()
        elif self.loss_type == "ldam":
            return LDAMLoss(self.kwargs.get("class_counts", []))
        else:
            raise ValueError(f"Nieznany typ funkcji straty: {self.loss_type}")


class StandardCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, meta=None):
        return self.loss_fn(outputs, targets)


class SampleWeightedCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_map = None

    def _build_weight_map(self, populacje, wieki):
        counts = Counter(zip(map(int, populacje), map(int, wieki)))
        total = sum(counts.values())
        raw_map = {k: total / v for k, v in counts.items()}
        mean_weight = sum(raw_map.values()) / len(raw_map)
        self.weight_map = {k: v / mean_weight for k, v in raw_map.items()}

    def forward(self, outputs, targets, meta=None):
        loss = F.cross_entropy(outputs, targets, reduction='none')

        if meta is None:
            return loss.mean()

        if self.weight_map is None:
            self._build_weight_map(meta["populacja"], meta["wiek"])

        weights = [self.weight_map.get((int(p), int(w)), 1.0)
                   for p, w in zip(meta["populacja"], meta["wiek"])]
        weights = torch.tensor(weights, dtype=torch.float32).to(outputs.device)

        return (loss * weights).mean()


class WeightedAgeCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, meta=None):
        loss = F.cross_entropy(outputs, targets, reduction='none')
        if meta is None:
            return loss.mean()

        # Oblicz częstość występowania każdego wieku
        wiek_tensor = meta['wiek'].clone().detach().to(dtype=torch.int64)
        unique_ages, counts = torch.unique(wiek_tensor, return_counts=True)
        freq_dict = {int(age.item()): count.item() for age, count in zip(unique_ages, counts)}

        # Zamień częstości na wagi odwrotne (rzadsze → większa waga)
        weights = [1.0 / freq_dict.get(int(w), 1.0) for w in meta['wiek']]
        weights = torch.tensor(weights, dtype=torch.float32).to(outputs.device)

        # Normalizacja wag do średniej 1.0 (dla stabilizacji)
        weights = weights * (len(weights) / weights.sum())

        return (loss * weights).mean()



class FocalLossWithAgeBoost(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, outputs, targets, meta=None):
        log_probs = F.log_softmax(outputs, dim=1)
        probs = torch.exp(log_probs)
        targets_onehot = F.one_hot(targets, num_classes=outputs.size(1)).float()
        pt = (probs * targets_onehot).sum(dim=1)
        loss = -((1 - pt) ** self.gamma) * torch.log(pt + 1e-8)

        if meta is not None:
            age_boost = {3: 2.0, 4: 2.0, 5: 2.0, 6: 2.0}
            weights = [age_boost.get(int(w), 1.0) for w in meta['wiek']]
            weights = torch.tensor(weights, dtype=torch.float32).to(outputs.device)
            loss = loss * weights

        return loss.mean()


class LDAMLoss(nn.Module):
    def __init__(self, class_counts, max_m=0.5, s=30):
        super().__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(class_counts).float() + 1e-8))
        m_list = m_list * (max_m / m_list.max())
        self.m_list = m_list
        self.s = s

    def forward(self, outputs, targets, meta=None):
        index = torch.zeros_like(outputs, dtype=torch.bool)
        index.scatter_(1, targets.data.view(-1, 1), 1)

        batch_m = torch.zeros_like(outputs).to(outputs.device)
        batch_m[index] = self.m_list[targets].to(outputs.device)

        outputs_m = outputs - batch_m
        return F.cross_entropy(self.s * outputs_m, targets)
