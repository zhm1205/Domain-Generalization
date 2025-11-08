"""Flexible builders for GRAPE (re-identification / segmentation / VF regression / UE)."""

from __future__ import annotations
from torch.utils.data import Dataset

from ..registry import register_dataset_builder
from ..utils.config import require_config, get_config
from omegaconf import DictConfig, OmegaConf

from .base_builder import BaseDatasetBuilder, BaseUEBuilder, _attach_dataset_task
from .grape import (
    GrapeReIDDataset,
    GrapeSegDataset,
    GrapeVFRegDataset,
    GrapeUEDataset,
)
from .transforms import get_transforms, get_seg_transforms

# ------------------------------- GRAPE Builder ----------------------------- #

class GrapeBuilder(BaseDatasetBuilder):
    """
    Unified builder for GRAPE.
    - dataset.task ∈ {'reid','seg','vf_reg','ue'}
    - 与 MIMICCXRBuilder 相同的构建逻辑与 DataLoader 行为
    """

    def __init__(self, config: DictConfig):
        dcfg: DictConfig = require_config(config, "dataset")
        self.task: str = get_config(dcfg, "task", "reid")
        super().__init__(config)

        # dataset-specific configuration
        self.csv_paths = {
            "train": get_config(dcfg, "train_csv_path"),
            "val":   get_config(dcfg, "val_csv_path"),
            "test":  get_config(dcfg, "test_csv_path"),
        }
        if None in self.csv_paths.values():
            raise ValueError("[GRAPE] CSV paths must be provided for train/val/test")

        # VF 列（可选）
        self.vf_cols = list(get_config(dcfg, "vf_cols", []))

    def default_sampler_name(self) -> str:
        return "pk" if self.task == "reid" else "random"

    # --- Internal utilities --- #
    def _dataset_cls(self):
        if self.task == "reid":
            return GrapeReIDDataset
        if self.task == "seg":
            return GrapeSegDataset
        if self.task == "vf_reg":
            return GrapeVFRegDataset
        if self.task == "ue":
            return GrapeUEDataset
        raise ValueError(f"[GRAPE] Unknown task type: {self.task}")

    @staticmethod
    def _trans_key_for_split(split: str) -> str:
        return "train" if split == "train" else "val"

    # --- Required overrides --- #
    def build_dataset(self, split: str, **overrides) -> Dataset:
        split = self._normalize_split(split)
        csv_path = overrides.get("csv_path", self.csv_paths.get(split))
        if csv_path is None:
            raise ValueError(f"[GRAPE] No CSV path for split '{split}'.")

        # ----- transform resolution（复用 MIMIC 配置键） -----
        t_trans_cfg = get_config(self.config, "training.data.transforms", OmegaConf.create({}))
        img_size = tuple(get_config(t_trans_cfg, "image_size", (256, 256)))
        mean_default = tuple(get_config(t_trans_cfg, "mean", (0.485, 0.456, 0.406)))
        std_default  = tuple(get_config(t_trans_cfg, "std",  (0.229, 0.224, 0.225)))
        forbid_geom  = bool(get_config(t_trans_cfg, "forbid_geom_aug", False))
        normalize_default = bool(require_config(t_trans_cfg, "normalize"))
        pixel_aug_default = bool(get_config(t_trans_cfg, "pixel_aug", True))
        
        is_train = (split == "train")
        geom_aug_default = (not forbid_geom) if is_train else False

        # per-call overrides
        normalize = overrides.get("normalize", normalize_default)
        pixel_aug = overrides.get("pixel_aug", pixel_aug_default)
        if split == "test":
            normalize = True
        geom_aug  = overrides.get("geom_aug", geom_aug_default)
        mean      = tuple(overrides.get("mean", mean_default))
        std       = tuple(overrides.get("std", std_default))

        if "transform" in overrides and overrides["transform"] is not None:
            transform = overrides["transform"]
        else:
            if self.task == "seg":
                transform = get_seg_transforms(
                    split=split,
                    image_size=img_size,
                    normalize=normalize,
                    geom_aug=geom_aug,
                    pixel_aug=pixel_aug,
                    mean=mean,
                    std=std,
                )
            else:
                transform = get_transforms(
                    split=split,
                    image_size=img_size,
                    normalize=normalize,
                    geom_aug=geom_aug,
                    pixel_aug=pixel_aug,
                    mean=mean,
                    std=std,
                )

        DS = self._dataset_cls()
        # 透传除去 transform/csv_path/图像增强键以外的参数
        ds = DS(
            csv_path=csv_path,
            split=split,
            transform=transform,
            vf_cols=(overrides.get("vf_cols", self.vf_cols) or None),
            **{k: v for k, v in overrides.items()
               if k not in {"csv_path", "transform", "image_size", "normalize", "geom_aug", "mean", "std", "vf_cols"}}
        )
        return ds


# ------------------------------ Registration ------------------------------ #
@register_dataset_builder("grape_seg")
class GrapeSegBuilder(GrapeBuilder):
    def __init__(self, config: DictConfig):
        cfg = _attach_dataset_task(config, "seg")
        super().__init__(cfg)

@register_dataset_builder("grape_vf_reg")
class GrapeVFRegBuilder(GrapeBuilder):
    def __init__(self, config: DictConfig):
        cfg = _attach_dataset_task(config, "vf_reg")
        super().__init__(cfg)

@register_dataset_builder("grape_reid")
class GrapeReIDBuilder(GrapeBuilder):
    def __init__(self, config: DictConfig):
        cfg = _attach_dataset_task(config, "reid")
        super().__init__(cfg)

register_dataset_builder("grape")(GrapeBuilder)

@register_dataset_builder("grape_ue")
class GrapeUEBuilder(BaseUEBuilder):
    """UE builder for GRAPE：沿用 BaseUEBuilder 策略"""
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._base_builder_name = "grape"
