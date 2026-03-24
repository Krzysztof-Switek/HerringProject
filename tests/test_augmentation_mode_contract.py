from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.data_loader.dataset import HerringDataset


@pytest.fixture
def cfg():
    return OmegaConf.create({
        "data": {
            "metadata_file": "tests/fixtures/metadata.xlsx",
            "root_dir": "tests/fixtures/data",
            "batch_size": 2,
            "active_populations": [1, 2],
        },
        "base_model": {
            "base_model": "resnet50",
            "image_size": 224,
            "pretrained": False,
            "freeze_encoder": False,
            "dropout_rate": 0.5,
            "weight_decay": 0.01,
        },
        "multitask_model": {
            "use": False,
            "backbone_model": {
                "model_name": "resnet50",
                "image_size": 224,
                "pretrained": False,
                "freeze_encoder": False,
                "dropout_rate": 0.5,
                "weight_decay": 0.01,
            },
        },
        "augmentation": {
            "mode": "strong",
            "rotation": 30,
            "crop_scale": [0.8, 1.0],
            "hflip_prob": 0.5,
            "vflip_prob": 0.5,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.05,
            "affine_degrees": 15,
            "affine_translate": [0.1, 0.1],
            "affine_scale": [0.9, 1.1],
            "affine_shear": 10,
            "gaussian_blur_kernel": 3,
        },
        "training": {
            "checkpoint_dir": "checkpoints",
        },
        "prediction": {
            "model_path": "checkpoints/model.pth",
        },
    })


def test_dataset_uses_strong_transform_when_mode_is_strong(cfg):
    dummy_metadata = {
        "a.jpg": (1, 2),
        "b.jpg": (2, 3),
    }

    dataset = HerringDataset(
        cfg,
        metadata_override=dummy_metadata,
        skip_validation=True,
    )

    assert dataset._get_train_transform() is dataset.train_transform_strong


def test_dataset_uses_base_transform_when_mode_is_base(cfg):
    dummy_metadata = {
        "a.jpg": (1, 2),
        "b.jpg": (2, 3),
    }

    cfg.augmentation.mode = "base"

    dataset = HerringDataset(
        cfg,
        metadata_override=dummy_metadata,
        skip_validation=True,
    )

    assert dataset._get_train_transform() is dataset.train_transform_base