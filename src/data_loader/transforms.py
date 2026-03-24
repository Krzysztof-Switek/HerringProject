from __future__ import annotations

from torchvision import transforms

from ..utils.config_helpers import get_active_image_size, get_augmentation_mode


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_image_size(cfg) -> int:
    return get_active_image_size(cfg)


def get_train_transform(cfg, mode: str | None = None):
    size = get_image_size(cfg)
    aug = cfg.augmentation

    resolved_mode = (mode or get_augmentation_mode(cfg)).lower()

    if resolved_mode == "base":
        return transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(p=float(aug.hflip_prob)),
            transforms.RandomVerticalFlip(p=float(aug.vflip_prob)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    if resolved_mode == "strong":
        return transforms.Compose([
            transforms.Resize(size),
            transforms.RandomResizedCrop(
                size,
                scale=tuple(float(x) for x in aug.crop_scale),
            ),
            transforms.RandomRotation(float(aug.rotation)),
            transforms.RandomHorizontalFlip(p=float(aug.hflip_prob)),
            transforms.RandomVerticalFlip(p=float(aug.vflip_prob)),
            transforms.ColorJitter(
                brightness=float(aug.brightness),
                contrast=float(aug.contrast),
                saturation=float(aug.saturation),
                hue=float(aug.hue),
            ),
            transforms.RandomAffine(
                degrees=float(aug.affine_degrees),
                translate=tuple(float(x) for x in aug.affine_translate),
                scale=tuple(float(x) for x in aug.affine_scale),
                shear=float(aug.affine_shear),
            ),
            transforms.GaussianBlur(kernel_size=int(aug.gaussian_blur_kernel)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    raise ValueError(f"Nieobsługiwany tryb augmentacji: {resolved_mode!r}")


def get_eval_transform(cfg):
    size = get_image_size(cfg)
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
