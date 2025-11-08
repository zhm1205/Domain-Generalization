"""Flexible builders for CSAW (classification / segmentation / re-identification / UE)."""

from __future__ import annotations
from torch.utils.data import Dataset

from ..registry import register_dataset_builder
from ..utils.config import require_config, get_config
from omegaconf import DictConfig, OmegaConf

from .base_builder import BaseDatasetBuilder, BaseUEBuilder, _attach_dataset_task
from .csaw import (
    CSAWReIDDataset,
    CSAWSegDataset,
    CSAWClassificationDataset,
    CSAWUEDataset,
)
from .transforms import get_transforms, get_seg_transforms


class CSAWBuilder(BaseDatasetBuilder):
    """
    Unified builder for CSAW.
    - dataset.task ∈ {'classification','seg','reid','ue'}
    - 读取根目录的 train.csv / val.csv / test.csv
    - 512 分辨率优先
    - 分类默认只加载 cls_use==1 的样本
    """

    def __init__(self, config: DictConfig):
        dcfg: DictConfig = require_config(config, "dataset")
        self.task: str = get_config(dcfg, "task", "classification")
        super().__init__(config)

        # dataset-specific configuration
        self.csv_paths = {
            "train": get_config(dcfg, "train_csv_path"),
            "val":   get_config(dcfg, "val_csv_path"),
            "test":  get_config(dcfg, "test_csv_path"),
        }
        if None in self.csv_paths.values():
            raise ValueError("[CSAW] CSV paths must be provided for train/val/test")

        # optional flags
        self.prefer_size = get_config(dcfg, "prefer_size", "512")
        self.require_mask = bool(get_config(dcfg, "require_mask", True))
        self.use_cls_balanced_only = bool(get_config(dcfg, "use_cls_balanced_only", True))
        self.check_files = bool(get_config(dcfg, "check_files", False))

    def default_sampler_name(self) -> str:
        return "pk" if self.task == "reid" else "random"

    def _dataset_cls(self):
        if self.task == "classification":
            return CSAWClassificationDataset
        if self.task == "seg":
            return CSAWSegDataset
        if self.task == "reid":
            return CSAWReIDDataset
        if self.task == "ue":
            return CSAWUEDataset
        raise ValueError(f"[CSAW] Unknown task type: {self.task}")

    @staticmethod
    def _trans_key_for_split(split: str) -> str:
        return "train" if split == "train" else "val"

    def build_dataset(self, split: str, **overrides) -> Dataset:
        split = self._normalize_split(split)
        csv_path = overrides.get("csv_path", self.csv_paths.get(split))
        if csv_path is None:
            raise ValueError(f"[CSAW] No CSV path for split '{split}'.")

        # -------- Transform resolution（复用你的配置键） --------
        t_trans_cfg = get_config(self.config, "training.data.transforms", OmegaConf.create({}))
        img_size = tuple(get_config(t_trans_cfg, "image_size", (256, 256)))
        mean_default = tuple(get_config(t_trans_cfg, "mean", (0.485, 0.456, 0.406)))
        std_default  = tuple(get_config(t_trans_cfg, "std",  (0.229, 0.224, 0.225)))
        forbid_geom  = bool(get_config(t_trans_cfg, "forbid_geom_aug", False))
        normalize_default = bool(require_config(t_trans_cfg, "normalize"))
        pixel_aug_default = bool(get_config(t_trans_cfg, "pixel_aug", True))

        is_train = (split == "train")
        geom_aug_default = (not forbid_geom) if is_train else False
        pixel_aug_default = (not forbid_geom) if is_train else False

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
        ds = DS(
            csv_path=csv_path,
            split=split,
            transform=transform,
            prefer_size=overrides.get("prefer_size", self.prefer_size),
            require_mask=overrides.get("require_mask", self.require_mask),
            use_cls_balanced_only=overrides.get("use_cls_balanced_only", self.use_cls_balanced_only),
            check_files=overrides.get("check_files", self.check_files),
            **{k: v for k, v in overrides.items()
               if k not in {"csv_path","transform","image_size","normalize","geom_aug","mean","std",
                            "prefer_size","require_mask","use_cls_balanced_only","check_files"}}
        )
        return ds


# ------------------------------ Registration ------------------------------ #

@register_dataset_builder("csaw")
class _CSAWMain(CSAWBuilder):
    def __init__(self, config: DictConfig):
        super().__init__(config)

@register_dataset_builder("csaw_reid")
class _CSAWReID(CSAWBuilder):
    def __init__(self, config: DictConfig):
        cfg = _attach_dataset_task(config, "reid")
        super().__init__(cfg)

@register_dataset_builder("csaw_seg")
class _CSAWSeg(CSAWBuilder):
    def __init__(self, config: DictConfig):
        cfg = _attach_dataset_task(config, "seg")
        super().__init__(cfg)

@register_dataset_builder("csaw_cls")
class _CSAWCls(CSAWBuilder):
    def __init__(self, config: DictConfig):
        cfg = _attach_dataset_task(config, "classification")
        super().__init__(cfg)

@register_dataset_builder("csaw_ue")
class CSAWUEBuilder(BaseUEBuilder):
    """UE builder for CSAW"""
    def __init__(self, config: DictConfig):
        cfg = _attach_dataset_task(config, "ue")
        super().__init__(cfg)
        self._base_builder_name = "csaw"
