from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.engine.loss_utills import LossFactory
from src.engine.trainer import Trainer
from src.engine.trainer_setup import run_training_loop
from src.utils.path_manager import PathManager


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "config.yaml"


def _load_cfg():
    return OmegaConf.load(CONFIG_PATH)


def _has_required_integration_data(cfg) -> tuple[bool, str]:
    path_manager = PathManager(PROJECT_ROOT, cfg)

    data_root = path_manager.data_root()
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    metadata_file = path_manager.metadata_file()

    if not train_dir.exists():
        return False, f"Brak katalogu integracyjnego: {train_dir}"
    if not val_dir.exists():
        return False, f"Brak katalogu integracyjnego: {val_dir}"
    if not metadata_file.exists():
        return False, f"Brak pliku metadata: {metadata_file}"

    if not any(train_dir.iterdir()):
        return False, f"Katalog train istnieje, ale jest pusty: {train_dir}"
    if not any(val_dir.iterdir()):
        return False, f"Katalog val istnieje, ale jest pusty: {val_dir}"

    return True, ""


@pytest.mark.parametrize("loss_name", _load_cfg().training.loss_type)
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
        "populacja": torch.randint(1, 3, (batch_size,)),
    }

    class_counts = [100, 80]
    class_freq = {0: 100, 1: 80}

    loss_factory = LossFactory(
        loss_type=loss_name,
        class_counts=class_counts if loss_name in ["ldam", "seesaw"] else None,
        class_freq=class_freq if loss_name == "class_balanced_focal" else None,
    )

    criterion = loss_factory.get()
    loss_value = criterion(logits, targets, meta)
    assert loss_value.item() > 0, f"Strata dla {loss_name} nie została poprawnie obliczona."


def test_training_pipeline_single_epoch(tmp_path):
    """
    Testuje cały pipeline treningowy dla jednej funkcji straty.
    Jest to test integracyjny: wymaga realnych danych wskazanych przez config.
    Jeśli danych brak, test jest pomijany zamiast fałszywie zgłaszać błąd logiki.
    """
    cfg = _load_cfg()

    has_data, reason = _has_required_integration_data(cfg)
    if not has_data:
        pytest.skip(f"Pomijam test integracyjny pipeline'u: {reason}")

    cfg.training.loss_type = ["standard_ce"]
    cfg.training.stop_after_one_epoch = True

    temp_config_path = tmp_path / "config_pipeline_test.yaml"
    OmegaConf.save(cfg, temp_config_path)

    trainer = Trainer(
        project_root=PROJECT_ROOT,
        config_path=temp_config_path,
    )

    run_training_loop(trainer)

    assert hasattr(trainer, "log_dir"), "Trainer nie ustawił log_dir."
    assert trainer.log_dir.exists(), "Nie utworzono katalogu logów."
    assert trainer.last_model_path is not None, "Nie zapisano żadnego modelu."


if __name__ == "__main__":
    import sys

    pytest_args = ["-v", __file__]
    sys.exit(pytest.main(pytest_args))