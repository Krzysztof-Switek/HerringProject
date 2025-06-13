import pytest
import torch
import sys
from pathlib import Path
from omegaconf import OmegaConf
from engine.loss_utills import LossFactory
from src.engine.trainer_setup import run_training_loop
from src.engine.trainer import Trainer

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

# === TEST KONFIGURACJI ===
project_root = Path(__file__).resolve().parent.parent / "src"
config_path = project_root / "config" / "config.yaml"
cfg = OmegaConf.load(config_path)

# === TESTY ===
@pytest.mark.parametrize("loss_name", cfg.training.loss_type)
def test_loss_function_compatibility(loss_name):
    """
    Testuje każdą funkcję straty zdefiniowaną w config.yaml,
    aby upewnić się, że może być utworzona i użyta do obliczenia straty.
    """
    batch_size = 4
    num_classes = 2
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,))
    meta = {
        "wiek": torch.randint(3, 9, (batch_size,)),
        "populacja": torch.randint(1, 3, (batch_size,))
    }

    class_counts = [100, 80]  # dummy counts
    class_freq = {0: 100, 1: 80}

    loss_factory = LossFactory(
        loss_type=loss_name,
        class_counts=class_counts if loss_name in ["ldam", "seesaw"] else None,
        class_freq=class_freq if loss_name == "class_balanced_focal" else None,
    )

    criterion = loss_factory.get()
    loss_value = criterion(logits, targets, meta)
    assert loss_value.item() > 0, f"Strata dla {loss_name} nie została poprawnie obliczona."


# === TEST URUCHOMIENIA CAŁEGO TRENERA (1 EPOKA, PIPELINE) ===
def test_training_pipeline_single_epoch():
    """
    Testuje cały pipeline treningowy dla jednej funkcji straty.
    """
    cfg.training.loss_type = ["standard_ce"]
    cfg.training.stop_after_one_epoch = True
    trainer = Trainer(project_root=Path(__file__).resolve().parent.parent, config_path=config_path)
    run_training_loop(trainer)
    assert trainer.log_dir.exists(), "Nie utworzono katalogu logów."
    assert trainer.last_model_path is not None, "Nie zapisano żadnego modelu."


if __name__ == "__main__":
    import sys
    pytest_args = ["-v", __file__]
    sys.exit(pytest.main(pytest_args))
