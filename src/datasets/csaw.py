from __future__ import annotations
import os
from typing import Optional, Callable, Literal, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import _to_tensor
from ..utils.logger import get_logger


def _pil_to_np_rgb(path: str) -> np.ndarray:
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception as e:
        raise RuntimeError(f"[CSAW] Failed to load image: {path} | {e}")

def _pil_to_np_mask(path: str) -> np.ndarray:
    try:
        m = Image.open(path).convert("L")
        # CSAW 二值：0/255；这里保留原始 0..255，交给 transform 再处理或训练时再阈值
        return np.array(m, dtype=np.uint8)
    except Exception as e:
        raise RuntimeError(f"[CSAW] Failed to load mask: {path} | {e}")


class _CSAWBase(Dataset):
    """
    CSAW base dataset with column normalization.

    期望 CSV（来自你的“根目录 train/val/test.csv”）包含：
      subject_id, study_id, dicom_id, split,
      image_path_512, image_path_1024,
      mask_bin_512,  mask_bin_1024,
      Laterality, ViewPosition,
      cls_visible, cls_use
    其余列（orig_h, orig_w, ...）若存在会被原样带到样本字典里（meta）。
    """

    def __init__(
        self,
        csv_path: str,
        split: Literal["train", "val", "validate", "test"] = "train",
        transform: Optional[Callable[..., Any]] = None,
        task_type: Literal["reid", "seg", "classification", "ue"] = "reid",
        prefer_size: Literal["512", "1024"] = "512",
        require_mask: bool = True,     # 仅对 seg 生效
        use_cls_balanced_only: bool = True,  # 仅对 classification 生效：只用 cls_use==1
        check_files: bool = False,     # 初始化时检查文件存在性
        **kwargs,
    ):
        super().__init__()
        self.logger = get_logger()
        self.csv_path = csv_path
        self.split = "val" if split == "validate" else split
        self.transform = transform
        self.task_type = task_type
        self.prefer_size = prefer_size
        self.require_mask = require_mask
        self.use_cls_balanced_only = use_cls_balanced_only
        self.check_files = check_files

        df = pd.read_csv(csv_path)
        self.df = self._normalize_columns(df)

        # 任务特定预处理
        if task_type == "reid":
            self._setup_reid()
        elif task_type == "seg":
            self._setup_seg()
        elif task_type == "classification":
            self._setup_cls()
        elif task_type == "ue":
            self._setup_ue()
        else:
            raise ValueError(f"[CSAW] Unknown task_type: {task_type}")

        self.logger.info(
            f"[CSAW] [{self.task_type}] Loaded {self.split}: {len(self.df)} rows, "
            f"{self.df['subject_id'].nunique()} subjects"
        )

    # ---------------- Column normalization ---------------- #

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        must_have = ["subject_id", "study_id", "split"]
        for c in must_have:
            if c not in df.columns:
                raise ValueError(f"[CSAW] CSV missing required column: {c}")

        # 选择图像/掩膜列
        img_col = f"image_path_{self.prefer_size}"
        alt_img_col = "image_path_1024" if self.prefer_size == "512" else "image_path_512"
        if img_col not in df.columns and alt_img_col in df.columns:
            self.logger.warning(f"[CSAW] Column '{img_col}' not in CSV; fallback to '{alt_img_col}'.")
            img_col = alt_img_col
        if img_col not in df.columns:
            raise ValueError(f"[CSAW] CSV missing image path column (tried: image_path_512/1024).")

        df = df.copy()
        df["image_path"] = df[img_col].astype(str)

        # 可选掩膜列（分割任务才会强校验）
        if "mask_bin_512" in df.columns or "mask_bin_1024" in df.columns:
            m_col = f"mask_bin_{self.prefer_size}"
            alt_m_col = "mask_bin_1024" if self.prefer_size == "512" else "mask_bin_512"
            if m_col not in df.columns and alt_m_col in df.columns:
                self.logger.warning(f"[CSAW] Column '{m_col}' not in CSV; fallback to '{alt_m_col}'.")
                m_col = alt_m_col
            df["mask_path"] = df.get(m_col, "").astype(str)

        # 视角统一为 view_position（小写下划线）
        if "ViewPosition" in df.columns:
            df["view_position"] = df["ViewPosition"].astype(str)
        elif "view_position" not in df.columns:
            df["view_position"] = "NA"

        # dicom_id 可缺省
        if "dicom_id" not in df.columns:
            # 用文件名 stem 做一个回退
            df["dicom_id"] = df["image_path"].apply(lambda p: os.path.splitext(os.path.basename(str(p)))[0])

        # 分类列
        if "cls_visible" not in df.columns:
            df["cls_visible"] = 0
        if "cls_use" not in df.columns:
            df["cls_use"] = 0

        # split 归一
        df["split"] = df["split"].replace({"validate": "val"})

        # 可选文件存在性检查
        if self.check_files:
            miss_img = (~df["image_path"].astype(str).apply(os.path.exists)).sum()
            if miss_img:
                self.logger.warning(f"[CSAW] Missing images: {miss_img}")
            if "mask_path" in df.columns:
                miss_m = (df["mask_path"].astype(str).eq("") | ~df["mask_path"].astype(str).apply(os.path.exists)).sum()
                self.logger.info(f"[CSAW] Mask availability: {len(df) - miss_m}/{len(df)}")

        return df.reset_index(drop=True)

    # ---------------- Task setups ---------------- #

    def _setup_reid(self):
        uniq = sorted(self.df["subject_id"].unique())
        self.subject_to_label = {sid: i for i, sid in enumerate(uniq)}
        self.label_to_subject = {i: sid for sid, i in self.subject_to_label.items()}
        self.df["reid_label"] = self.df["subject_id"].map(self.subject_to_label).astype(int)

        cnt = self.df["reid_label"].value_counts()
        self.num_classes = len(uniq)
        self.logger.info(f"[CSAW:ReID] classes={self.num_classes}, imgs/subject "
                         f"min={int(cnt.min())}, max={int(cnt.max())}, mean={cnt.mean():.2f}")

    def _setup_seg(self):
        if "mask_path" not in self.df.columns:
            raise ValueError("[CSAW:Seg] 'mask_path' missing in CSV (mask_bin_512/1024).")
        if self.require_mask:
            keep = self.df["mask_path"].astype(str).str.len() > 0
            n0 = len(self.df)
            self.df = self.df[keep].reset_index(drop=True)
            self.logger.info(f"[CSAW:Seg] require_mask=True, kept {len(self.df)}/{n0} rows with masks.")
            if len(self.df) == 0:
                raise ValueError("[CSAW:Seg] No rows with masks after filtering.")

    def _setup_cls(self):
        if "cls_visible" not in self.df.columns:
            raise ValueError("[CSAW:Cls] 'cls_visible' missing in CSV.")
        if self.use_cls_balanced_only:
            if "cls_use" not in self.df.columns:
                raise ValueError("[CSAW:Cls] 'cls_use' missing but use_cls_balanced_only=True.")
            n0 = len(self.df)
            self.df = self.df[self.df["cls_use"] == 1].reset_index(drop=True)
            self.logger.info(f"[CSAW:Cls] use_cls_balanced_only=True, kept {len(self.df)}/{n0} rows.")

    def _setup_ue(self):
        uniq = sorted(self.df["subject_id"].unique())
        self.ue_subject_to_label = {sid: i for i, sid in enumerate(uniq)}
        self.df["reid_label"] = self.df["subject_id"].map(self.ue_subject_to_label).astype(int)
        self.num_classes_reid = len(uniq)

    # ---------------- std API ---------------- #

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, do_transform=True):
        row = self.df.iloc[idx]
        img_np = _pil_to_np_rgb(row["image_path"])

        if self.task_type == "seg":
            mask_np = _pil_to_np_mask(row["mask_path"])
            if self.transform and do_transform:
                image, mask_t = self.transform(img_np, mask_np)
            else:
                image = _to_tensor(img_np)
                mask_t = torch.as_tensor(mask_np).long()
            return self._item_seg(row, image, mask_t)

        # image-only tasks
        if self.transform and do_transform:
            image = self.transform(img_np)
        else:
            image = _to_tensor(img_np)

        if self.task_type == "reid":
            return self._item_reid(row, image)
        if self.task_type == "classification":
            return self._item_cls(row, image)
        if self.task_type == "ue":
            return self._item_ue(row, image)
        raise RuntimeError("Unreachable")

    # ---------------- item builders ---------------- #

    def _basic_meta(self, row) -> dict:
        return {
            "subject_id": row["subject_id"],
            "dicom_id": row.get("dicom_id", os.path.splitext(os.path.basename(row["image_path"]))[0]),
            "study_id": row["study_id"],
            "image_path": row["image_path"],
        }

    def _item_reid(self, row, image):
        meta = self._basic_meta(row)
        out = {
            "image": image,
            "label": int(row["reid_label"]),
            "view_position": row.get("view_position", "NA"),
            **meta,
        }
        return out

    def _item_seg(self, row, image, mask_t: torch.Tensor):
        meta = self._basic_meta(row)
        return {
            "image": image,
            "mask": mask_t,   # uint8/long, 二值或后续映射
            **meta,
        }

    def _item_cls(self, row, image):
        meta = self._basic_meta(row)
        return {
            "image": image,
            "label": int(row["cls_visible"]),  # 0/1
            "cls_use": int(row.get("cls_use", 0)),
            **meta,
        }

    def _item_ue(self, row, image):
        meta = self._basic_meta(row)
        return {
            "image": image,
            "targets": {"reid": int(row["reid_label"])},
            "meta": meta,
        }

    # ---------------- sampler hook ---------------- #
    def labels_for_sampling(self, kind: str = "reid") -> torch.Tensor:
        assert kind == "reid", "Only 'reid' sampling is supported."
        return torch.as_tensor(self.df["reid_label"].values, dtype=torch.long)


# -------------------------- Task-Specific Classes -------------------------- #

class CSAWReIDDataset(_CSAWBase):
    def __init__(self, *args, **kwargs):
        kwargs["task_type"] = "reid"
        super().__init__(*args, **kwargs)

class CSAWSegDataset(_CSAWBase):
    def __init__(self, *args, **kwargs):
        kwargs["task_type"] = "seg"
        super().__init__(*args, **kwargs)

class CSAWClassificationDataset(_CSAWBase):
    def __init__(self, *args, **kwargs):
        kwargs["task_type"] = "classification"
        super().__init__(*args, **kwargs)

class CSAWUEDataset(_CSAWBase):
    def __init__(self, *args, **kwargs):
        kwargs["task_type"] = "ue"
        super().__init__(*args, **kwargs)
