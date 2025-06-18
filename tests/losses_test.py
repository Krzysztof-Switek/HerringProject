import pytest
import torch

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

# Stałe testowe dane
NUM_CLASSES = 2
BATCH_SIZE = 8
CLASS_COUNTS = [50, 200, 100]
CLASS_FREQ = {0: 50, 1: 200, 2: 100}

@pytest.mark.parametrize("loss_fn", [
    StandardCrossEntropy(),
    SampleWeightedCrossEntropy(),
    WeightedAgeCrossEntropy(),
    FocalLossWithAgeBoost(),
    LDAMLoss(CLASS_COUNTS),
    AsymmetricFocalLoss(),
    ClassBalancedFocalLoss(CLASS_FREQ),
    FocalTverskyLoss(),
    GHMLoss(),
    SeesawLoss(CLASS_COUNTS),
])
def test_loss_computation(loss_fn):
    torch.manual_seed(42)

    outputs = torch.randn(BATCH_SIZE, NUM_CLASSES, requires_grad=True)
    targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

    meta = {
        "populacja": torch.randint(1, 3, (BATCH_SIZE,)),
        "wiek": torch.randint(1, 11, (BATCH_SIZE,))
    }

    try:
        loss = loss_fn(outputs, targets, meta)
        assert not torch.isnan(loss), f"Loss {loss_fn.__class__.__name__} returned NaN"
        assert not torch.isinf(loss), f"Loss {loss_fn.__class__.__name__} returned Inf"

        loss.backward()  # test propagacji gradientu
        print(f"{loss_fn.__class__.__name__} działa poprawnie: loss={loss.item():.4f}")

    except Exception as e:
        pytest.fail(f"Funkcja {loss_fn.__class__.__name__} rzuciła wyjątek: {str(e)}")
