import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


class BaseMultitaskLossWrapper(nn.Module):
    def unwrap_logits(self, outputs):
        return outputs[0] if isinstance(outputs, tuple) else outputs


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
        elif self.loss_type == "asymmetric_focal":
            return AsymmetricFocalLoss()
        elif self.loss_type == "class_balanced_focal":
            return ClassBalancedFocalLoss(self.kwargs.get("class_freq", {}))
        elif self.loss_type == "focal_tversky":
            return FocalTverskyLoss()
        elif self.loss_type == "ghm":
            return GHMLoss()
        elif self.loss_type == "seesaw":
            return SeesawLoss(self.kwargs.get("class_counts", []))
        else:
            raise ValueError(f"Nieznany typ funkcji straty: {self.loss_type}")


class StandardCrossEntropy(BaseMultitaskLossWrapper):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, meta=None):
        outputs = self.unwrap_logits(outputs)
        return self.loss_fn(outputs, targets)


class SampleWeightedCrossEntropy(BaseMultitaskLossWrapper):
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
        outputs = self.unwrap_logits(outputs)
        loss = F.cross_entropy(outputs, targets, reduction='none')

        if meta is None:
            return loss.mean()

        if self.weight_map is None:
            self._build_weight_map(meta["populacja"], meta["wiek"])

        weights = [self.weight_map.get((int(p), int(w)), 1.0)
                   for p, w in zip(meta["populacja"], meta["wiek"])]
        weights = torch.tensor(weights, dtype=torch.float32).to(outputs.device)

        return (loss * weights).mean()


class WeightedAgeCrossEntropy(BaseMultitaskLossWrapper):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, meta=None):
        outputs = self.unwrap_logits(outputs)
        loss = F.cross_entropy(outputs, targets, reduction='none')
        if meta is None:
            return loss.mean()

        wiek_tensor = meta['wiek'].clone().detach().to(dtype=torch.int64)
        unique_ages, counts = torch.unique(wiek_tensor, return_counts=True)
        freq_dict = {int(age.item()): count.item() for age, count in zip(unique_ages, counts)}

        weights = [1.0 / freq_dict.get(int(w), 1.0) for w in meta['wiek']]
        weights = torch.tensor(weights, dtype=torch.float32).to(outputs.device)
        weights = weights * (len(weights) / weights.sum())

        return (loss * weights).mean()


class FocalLossWithAgeBoost(BaseMultitaskLossWrapper):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, outputs, targets, meta=None):
        outputs = self.unwrap_logits(outputs)
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


class LDAMLoss(BaseMultitaskLossWrapper):
    def __init__(self, class_counts, max_m=0.5, s=30):
        super().__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(class_counts).float() + 1e-8))
        m_list = m_list * (max_m / m_list.max())
        self.m_list = m_list
        self.s = s

    def forward(self, outputs, targets, meta=None):
        outputs = self.unwrap_logits(outputs)
        index = torch.zeros_like(outputs, dtype=torch.bool)
        index.scatter_(1, targets.data.view(-1, 1), 1)
        batch_m = torch.zeros_like(outputs).to(outputs.device)
        batch_m[index] = self.m_list[targets].to(outputs.device)
        outputs_m = outputs - batch_m
        return F.cross_entropy(self.s * outputs_m, targets)


class AsymmetricFocalLoss(BaseMultitaskLossWrapper):
    def __init__(self, gamma_pos=0.0, gamma_neg=2.0):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, outputs, targets, meta=None):
        outputs = self.unwrap_logits(outputs)
        log_probs = F.log_softmax(outputs, dim=1)
        probs = torch.exp(log_probs)
        targets_onehot = F.one_hot(targets, num_classes=outputs.size(1)).float()
        pt = (probs * targets_onehot).sum(dim=1)
        gamma = torch.where(targets == 1, self.gamma_pos, self.gamma_neg)
        loss = -((1 - pt) ** gamma) * torch.log(pt + 1e-8)
        return loss.mean()


class ClassBalancedFocalLoss(BaseMultitaskLossWrapper):
    def __init__(self, class_freq: dict, beta=0.999, gamma=2.0):
        super().__init__()
        self.class_freq = class_freq
        self.beta = beta
        self.gamma = gamma
        self.weights = self._compute_class_weights()

    def _compute_class_weights(self):
        weights = {}
        all_classes = list(range(max(self.class_freq.keys()) + 1))
        for cls in all_classes:
            n = self.class_freq.get(cls, 1)
            effective_num = 1.0 - self.beta ** n
            weight = (1.0 - self.beta) / (effective_num + 1e-8)
            weights[cls] = weight
        total = sum(weights.values())
        return {cls: w / total for cls, w in weights.items()}

    def forward(self, outputs, targets, meta=None):
        outputs = self.unwrap_logits(outputs)
        log_probs = F.log_softmax(outputs, dim=1)
        probs = torch.exp(log_probs)
        targets_onehot = F.one_hot(targets, num_classes=outputs.size(1)).float()
        pt = (probs * targets_onehot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma
        class_weights = torch.tensor([self.weights[int(t.item())] for t in targets], dtype=torch.float32).to(outputs.device)
        return -(focal_weight * class_weights * torch.log(pt + 1e-8)).mean()


class FocalTverskyLoss(BaseMultitaskLossWrapper):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, outputs, targets, meta=None):
        outputs = self.unwrap_logits(outputs)
        targets_onehot = F.one_hot(targets, num_classes=outputs.size(1)).float()
        probs = F.softmax(outputs, dim=1)
        TP = (probs * targets_onehot).sum(dim=0)
        FP = (probs * (1 - targets_onehot)).sum(dim=0)
        FN = ((1 - probs) * targets_onehot).sum(dim=0)
        tversky = (TP + 1e-8) / (TP + self.alpha * FP + self.beta * FN + 1e-8)
        loss = (1 - tversky) ** self.gamma
        return loss.mean()


class GHMLoss(BaseMultitaskLossWrapper):
    def __init__(self, bins=10, momentum=0.75):
        super().__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.linspace(0, 1, bins + 1)
        self.acc_sum = torch.zeros(bins)

    def forward(self, outputs, targets, meta=None):
        outputs = self.unwrap_logits(outputs)
        g = torch.abs(F.softmax(outputs, dim=1) - F.one_hot(targets, num_classes=outputs.size(1)).float()).sum(dim=1)
        total = outputs.size(0)
        weights = torch.zeros_like(g)
        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                self.acc_sum[i] = self.momentum * self.acc_sum[i] + (1 - self.momentum) * num_in_bin
                weights[inds] = total / self.acc_sum[i]
        loss = F.cross_entropy(outputs, targets, reduction='none')
        return (loss * weights).mean()


class SeesawLoss(BaseMultitaskLossWrapper):
    def __init__(self, class_counts, p=0.8):
        super().__init__()
        self.class_counts = torch.tensor(class_counts).float()
        self.p = p

    def forward(self, outputs, targets, meta=None):
        outputs = self.unwrap_logits(outputs)
        logits = F.log_softmax(outputs, dim=1)
        bias_weights = self.class_counts[targets] ** self.p
        loss = -logits[range(len(targets)), targets] * bias_weights.to(outputs.device)
        return loss.mean()

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, classification_loss, regression_loss, method="none", static_weights=None):
        super().__init__()
        self.classification_loss = classification_loss
        self.regression_loss = regression_loss
        self.method = method

        if method == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(2))  # log(sigma^2) dla każdej taski
        elif method == "static":
            if static_weights is None:
                raise ValueError("Wymagane static_weights, jeśli metoda to 'static'")
            self.static_weights = static_weights
        elif method == "gradnorm":
            self.alpha = 1.5  # zalecane w GradNorm
            self.initial_losses = None  # zostanie ustalone po 1 ep.
            self.weights = nn.Parameter(torch.ones(2))  # wagi do optymalizacji
        elif method == "none":
            pass
        else:
            raise NotImplementedError(f"Obsługa wag {self.method} jeszcze niezaimplementowana")

    def forward(self, outputs, targets, meta):
        logits, age_pred = outputs
        cls_loss = self.classification_loss(logits, targets, meta)
        reg_loss = self.regression_loss(age_pred.squeeze(), meta['wiek'].float())

        if self.method == "uncertainty":
            precision_cls = torch.exp(-self.log_vars[0])
            precision_reg = torch.exp(-self.log_vars[1])
            loss = precision_cls * cls_loss + self.log_vars[0] + precision_reg * reg_loss + self.log_vars[1]
            return loss

        elif self.method == "static":
            return self.static_weights['classification'] * cls_loss + \
                   self.static_weights['age'] * reg_loss

        elif self.method == "gradnorm":
            # Ustal straty z początku (1 epoka)
            if self.initial_losses is None:
                self.initial_losses = (cls_loss.detach(), reg_loss.detach())

            # Oblicz gradienty względem shared weights
            self.weights = nn.Parameter(F.relu(self.weights))
            loss_vec = torch.stack([cls_loss, reg_loss])
            weighted_loss = torch.sum(self.weights * loss_vec)
            weighted_loss.backward(retain_graph=True)

            # Oblicz gradienty normy
            grads = []
            for i, l in enumerate([cls_loss, reg_loss]):
                grad = torch.autograd.grad(l, self.weights, retain_graph=True, allow_unused=True)[0]
                grads.append(torch.norm(grad))

            # Oblicz współczynnik skalowania
            avg_loss_ratio = torch.tensor([
                (cls_loss / self.initial_losses[0]).item(),
                (reg_loss / self.initial_losses[1]).item()
            ])
            avg_grad = sum(grads) / 2
            target_grads = self.alpha * avg_loss_ratio * avg_grad

            loss_gradnorm = torch.sum(torch.abs(grads - target_grads.detach()))
            return weighted_loss + loss_gradnorm

        elif self.method == "none":
            return cls_loss + reg_loss

        else:
            raise NotImplementedError(f"Obsługa wag {self.method} jeszcze niezaimplementowana")

