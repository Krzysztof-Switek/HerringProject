import pytest
import torch
from src.data_loader.dataset import HerringDataset
from src.models.model import HerringModel
from src.engine.loss_utills import (
    StandardCrossEntropy,
    SampleWeightedCrossEntropy,
    WeightedAgeCrossEntropy,
    FocalLossWithAgeBoost,
    LDAMLoss,
    AsymmetricFocalLoss,
    ClassBalancedFocalLoss,
    FocalTverskyLoss,
    GHMLoss,
    SeesawLoss
)
from omegaconf import OmegaConf
from pathlib import Path


@pytest.fixture(scope="module")
def cfg():
    config_path = Path(__file__).parent.parent / "src" / "config" / "config.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.data.root_dir = str((Path(__file__).parent.parent / cfg.data.root_dir).resolve())
    return cfg


@pytest.mark.parametrize("loss_fn", [
    StandardCrossEntropy(),
    SampleWeightedCrossEntropy(),
    WeightedAgeCrossEntropy(),
    FocalLossWithAgeBoost(),
    LDAMLoss([50, 200, 100]),
    AsymmetricFocalLoss(),
    ClassBalancedFocalLoss({0: 50, 1: 200, 2: 100}),
    FocalTverskyLoss(),
    GHMLoss(),
    SeesawLoss([50, 200, 100])
])
def test_loss_integration_with_model(cfg, loss_fn):
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = HerringModel(cfg).to(device)
    model.train()

    # Dane
    dataset = HerringDataset(cfg)
    train_loader, _, _ = dataset.get_loaders()  # 🟡 ZMIANA: zamiast dataset.train_set
    inputs, targets, meta = next(iter(train_loader))

    inputs, targets = inputs.to(device), targets.to(device)
    for key in meta:
        meta[key] = meta[key].to(device)

    # Forward
    outputs = model(inputs)
    preds = outputs.argmax(dim=1)  # 🟡 ZMIANA: rozkład predykcji
    print(f"\n[{loss_fn.__class__.__name__}] Predykcje: {torch.bincount(preds).cpu().tolist()}")
    print(f"[{loss_fn.__class__.__name__}] Targety:    {torch.bincount(targets).cpu().tolist()}")

    # Sprawdź, czy model nie przewiduje zawsze jednej klasy
    assert (preds != preds[0]).any(), f"{loss_fn.__class__.__name__} zawsze przewiduje jedną klasę"  # 🟡 ZMIANA

    # Loss + backward
    loss = loss_fn(outputs, targets, meta)
    assert not torch.isnan(loss), f"{loss_fn.__class__.__name__} zwrócił NaN"
    assert not torch.isinf(loss), f"{loss_fn.__class__.__name__} zwrócił Inf"

    loss.backward()

    # Gradienty – podsumowanie
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            print(f"[{loss_fn.__class__.__name__}] Grad {name}: {norm:.6f}")  # 🟡 ZMIANA

    print(f"{loss_fn.__class__.__name__}: OK, loss = {loss.item():.4f}")
