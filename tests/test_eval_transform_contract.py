from __future__ import annotations

from omegaconf import OmegaConf
from torchvision import transforms

from src.data_loader.transforms import get_eval_transform


def _cfg_single_task(image_size: int = 224):
    return OmegaConf.create({
        "base_model": {
            "base_model": "resnet50",
            "image_size": image_size,
            "pretrained": False,
            "freeze_encoder": False,
            "dropout_rate": 0.5,
            "weight_decay": 0.01,
        },
        "multitask_model": {
            "use": False,
            "backbone_model": {
                "model_name": "resnet50",
                "image_size": image_size,
                "pretrained": False,
                "freeze_encoder": False,
                "dropout_rate": 0.5,
                "weight_decay": 0.01,
            },
        },
        "augmentation": {
            "mode": "strong",
        },
    })


def _cfg_multitask(image_size: int = 320):
    return OmegaConf.create({
        "base_model": {
            "base_model": "resnet50",
            "image_size": 224,
            "pretrained": False,
            "freeze_encoder": False,
            "dropout_rate": 0.5,
            "weight_decay": 0.01,
        },
        "multitask_model": {
            "use": True,
            "backbone_model": {
                "model_name": "efficientnet_b0",
                "image_size": image_size,
                "pretrained": False,
                "freeze_encoder": False,
                "dropout_rate": 0.5,
                "weight_decay": 0.01,
            },
        },
        "augmentation": {
            "mode": "strong",
        },
    })


def _normalize_size(value):
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise AssertionError(f"Nieobsługiwany format size: {value!r}")


def test_eval_transform_contains_required_ops_single_task():
    cfg = _cfg_single_task(image_size=224)

    transform = get_eval_transform(cfg)
    ops = transform.transforms

    assert isinstance(transform, transforms.Compose)
    assert isinstance(ops[0], transforms.Resize)
    assert isinstance(ops[1], transforms.CenterCrop)
    assert isinstance(ops[2], transforms.ToTensor)
    assert isinstance(ops[3], transforms.Normalize)


def test_eval_transform_uses_single_task_image_size():
    cfg = _cfg_single_task(image_size=224)

    transform = get_eval_transform(cfg)
    resize_op = transform.transforms[0]
    crop_op = transform.transforms[1]

    resize_h, resize_w = _normalize_size(resize_op.size)
    crop_h, crop_w = _normalize_size(crop_op.size)

    assert (resize_h, resize_w) == (224, 224)
    assert (crop_h, crop_w) == (224, 224)


def test_eval_transform_uses_multitask_backbone_image_size():
    cfg = _cfg_multitask(image_size=320)

    transform = get_eval_transform(cfg)
    resize_op = transform.transforms[0]
    crop_op = transform.transforms[1]

    resize_h, resize_w = _normalize_size(resize_op.size)
    crop_h, crop_w = _normalize_size(crop_op.size)

    assert (resize_h, resize_w) == (320, 320)
    assert (crop_h, crop_w) == (320, 320)