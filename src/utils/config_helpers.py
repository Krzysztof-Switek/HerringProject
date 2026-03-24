from __future__ import annotations

from typing import Any


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default

    if isinstance(cfg, dict):
        return cfg.get(key, default)

    if hasattr(cfg, key):
        return getattr(cfg, key)

    getter = getattr(cfg, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            return getter(key)

    return default


def is_multitask_enabled(cfg: Any) -> bool:
    multitask_cfg = _get(cfg, "multitask_model", None)
    if multitask_cfg is None:
        return False
    use_value = _get(multitask_cfg, "use", False)
    return bool(use_value)


def get_active_model_name(cfg: Any) -> str:
    if is_multitask_enabled(cfg):
        multitask_cfg = _get(cfg, "multitask_model")
        backbone_cfg = _get(multitask_cfg, "backbone_model")
        model_name = _get(backbone_cfg, "model_name", None)
        if not model_name:
            raise ValueError("Brakuje multitask_model.backbone_model.model_name w configu")
        return str(model_name)

    base_model_cfg = _get(cfg, "base_model")
    model_name = _get(base_model_cfg, "base_model", None)
    if not model_name:
        raise ValueError("Brakuje base_model.base_model w configu")
    return str(model_name)


def get_active_image_size(cfg: Any) -> int:
    if is_multitask_enabled(cfg):
        multitask_cfg = _get(cfg, "multitask_model")
        backbone_cfg = _get(multitask_cfg, "backbone_model")
        image_size = _get(backbone_cfg, "image_size", None)
        if image_size is None:
            raise ValueError("Brakuje multitask_model.backbone_model.image_size w configu")
        return int(image_size)

    base_model_cfg = _get(cfg, "base_model")
    image_size = _get(base_model_cfg, "image_size", None)
    if image_size is None:
        raise ValueError("Brakuje base_model.image_size w configu")
    return int(image_size)


def get_augmentation_mode(cfg: Any) -> str:
    augmentation_cfg = _get(cfg, "augmentation", None)
    mode = _get(augmentation_cfg, "mode", "strong")
    mode = str(mode).strip().lower()

    if mode not in {"base", "strong"}:
        raise ValueError(f"Nieobsługiwany augmentation.mode={mode!r}; dozwolone: 'base', 'strong'")

    return mode

