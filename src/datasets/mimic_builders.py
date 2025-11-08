# file: src/datasets/builders_mimic_cxr.py
"""Flexible builders for MIMIC-CXR (classification & re-identification)."""

from __future__ import annotations
from torch.utils.data import Dataset
from ..registry import register_dataset_builder, get_dataset_builder
from ..utils.config import require_config, get_config
from .mimic_cxr import (
    MIMICCXRReIDDataset,
    MIMICCXRClassificationDataset,
    MIMICCXRUEDataset,
)
from .transforms import get_transforms
from omegaconf import DictConfig, OmegaConf
from .base_builder import BaseDatasetBuilder, BaseUEBuilder, _attach_dataset_task

# ------------------------------- MIMIC Builder ----------------------------- #

class MIMICCXRBuilder(BaseDatasetBuilder):
    """
    Unified builder for MIMIC-CXR.
    - Set `config['dataset']['task']` to 'classification' or 'reid'.
    - Lazily builds datasets/loaders.
    - Supports per-call overrides (batch_size, num_workers, transform, image_size, etc.).
    """

    def __init__(self, config: DictConfig):
        dcfg: DictConfig = require_config(config, "dataset")
        self.task: str = get_config(dcfg, "task", "classification")
        super().__init__(config)

        # dataset-specific configuration
        self.csv_paths = {
            'train': get_config(dcfg, "train_csv_path"),
            'val':   get_config(dcfg, "val_csv_path"),
            'test':  get_config(dcfg, "test_csv_path"),
        }
        if None in self.csv_paths.values():
            raise ValueError("Dataset CSV paths must be provided for train/val/test")
    
    def default_sampler_name(self) -> str:
        return "pk" if self.task == "reid" else "random"

    # --- Internal utilities --- #
    def _dataset_cls(self):
        if self.task == "classification":
            return MIMICCXRClassificationDataset
        elif self.task == "reid":
            return MIMICCXRReIDDataset
        elif self.task == "ue":
            return MIMICCXRUEDataset
        else:
            raise ValueError(f"Unknown task type: {self.task}. Expected 'classification' or 'reid'.")

    @staticmethod
    def _trans_key_for_split(split: str) -> str:
        """Map split name to transform group: 'train' for training, 'val' otherwise."""
        return "train" if split == "train" else "val"

    # --- Required overrides --- #
    def build_dataset(self, split: str, **overrides) -> Dataset:
        split = self._normalize_split(split)
        csv_path = overrides.get("csv_path", self.csv_paths.get(split))
        if csv_path is None:
            raise ValueError(
                f"No CSV path found for split '{split}'. Available splits: {list(self.csv_paths.keys())}"
            )

        # ----------------------- Transform resolution ----------------------- #
        # We compute sensible defaults based on training.* config,
        # but still allow call-site overrides to take precedence.
        # [CHANGED] read transform-related defaults (mean/std/forbid_geom_aug)
        t_trans_cfg = get_config(self.config, "training.data.transforms", OmegaConf.create({}))

        img_size = tuple(get_config(t_trans_cfg,"image_size", (256, 256)))
        mean_default = tuple(get_config(t_trans_cfg, "mean", (0.485, 0.456, 0.406)))
        std_default  = tuple(get_config(t_trans_cfg, "std",  (0.229, 0.224, 0.225)))
        forbid_geom  = bool(get_config(t_trans_cfg, "forbid_geom_aug", False))
        normalize_default = bool(require_config(t_trans_cfg, "normalize"))
        pixel_aug_default = bool(get_config(t_trans_cfg, "pixel_aug", True))
        is_train = (split == "train")
        # [CHANGED] default flags for this split
        geom_aug_default  = (not forbid_geom) if is_train else False
        pixel_aug_default = (not forbid_geom) if is_train else False
        # [CHANGED] allow per-call overrides for fine control
        normalize = overrides.get("normalize", normalize_default)
        pixel_aug = overrides.get("pixel_aug", pixel_aug_default)
        if split == "test":
            normalize = True
        geom_aug  = overrides.get("geom_aug", geom_aug_default)
        mean      = tuple(overrides.get("mean", mean_default))
        std       = tuple(overrides.get("std", std_default))
        self.logger.info(f"Building dataset '{split}' with transforms: img_size={img_size}, normalize={normalize}, geom_aug={geom_aug}, mean={mean}, std={std}")
        # [CHANGED] only build a default transform when caller didn't supply one
        if "transform" in overrides and overrides["transform"] is not None:
            transform = overrides["transform"]
        else:
            transform = get_transforms(
                split=split,                  # pass real split: 'train' | 'val' | 'test'
                image_size=img_size,
                normalize=normalize,          # control Normalize inside pipeline (UE needs False on train)
                geom_aug=geom_aug,            # toggle geometry aug on train; always False for val/test
                pixel_aug=pixel_aug,          # toggle pixel aug on train; always False for val/test
                mean=mean,
                std=std,
            )

        DS = self._dataset_cls()
        return DS(
            csv_path=csv_path,
            split=split,
            transform=transform,
            **{k: v for k, v in overrides.items() if k not in {"csv_path", "transform", "image_size",
                                                               "normalize", "geom_aug", "mean", "std"}}  # [CHANGED]
        )


# ------------------------------ Registration ------------------------------ #


# Backward-compatible aliases that force a task by wrapping config
class _MIMICCXRReIDBuilderAlias(MIMICCXRBuilder):
    def __init__(self, config: DictConfig):
        cfg = _attach_dataset_task(config, "reid")
        super().__init__(cfg)

class _MIMICCXRClsBuilderAlias(MIMICCXRBuilder):
    def __init__(self, config: DictConfig):
        cfg = _attach_dataset_task(config, "classification")
        super().__init__(cfg)

# New unified name
register_dataset_builder("mimic_cxr")(MIMICCXRBuilder)
register_dataset_builder("mimic_cxr_reid")(_MIMICCXRReIDBuilderAlias)
register_dataset_builder("mimic_cxr_cls")(_MIMICCXRClsBuilderAlias)

@register_dataset_builder("mimic_cxr_ue")
class MIMICCXRUEBuilder(BaseUEBuilder):
    """
    Thin UE builder for MIMIC-CXR: delegate to BaseUEBuilder.
    If you want to force the base task builder name here, you can override __init__.
    """
    def __init__(self, config: DictConfig):
        cfg = _attach_dataset_task(config, "ue")
        super().__init__(cfg)
        # Optionally lock base builder name for this dataset family:
        self._base_builder_name = "mimic_cxr"