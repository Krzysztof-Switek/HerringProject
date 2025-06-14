import pytest
from src.data_loader.dataset import HerringDataset
from omegaconf import OmegaConf
from pathlib import Path
import torch


@pytest.fixture
def minimal_cfg():
    path = Path(__file__).parent.parent / "src" / "config" / "config.yaml"
    cfg = OmegaConf.load(path)
    return cfg


def test_herring_dataset_loaders(minimal_cfg):
    dataset = HerringDataset(minimal_cfg)
    train_loader, val_loader, classes = dataset.get_loaders()

    # Sprawdź, czy klasy są poprawne
    assert classes == ['1', '2'], f"Nieprawidłowe klasy: {classes}"

    # Sprawdź kilka próbek z train_loader
    batch = next(iter(train_loader))
    images, labels, metadata = batch

    # Sprawdź typy i kształty
    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(metadata, dict)
    assert "populacja" in metadata and "wiek" in metadata

    # Sprawdź wartości
    unique_labels = labels.unique().tolist()
    for label in unique_labels:
        assert label in [0, 1], f"Nieprawidłowa etykieta: {label}"

    # Weryfikacja shape obrazu
    _, c, h, w = images.shape
    expected_size = minimal_cfg.model.image_size
    assert c == 3
    assert h == expected_size and w == expected_size, f"Niepoprawny rozmiar obrazu: {h}x{w}"

    print("✅ Dataset działa poprawnie.")


