from __future__ import annotations
import os
from typing import Optional, Callable, Literal, Any, List, Tuple
from .transforms import _to_tensor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from ..utils.logger import get_logger


# ----------------------------- Common utilities ----------------------------- #

def _pil_to_np_rgb(path: str) -> np.ndarray:
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)  # HWC, uint8
    except Exception as e:
        raise RuntimeError(f"[GRAPE] Failed to load image: {path} | {e}")

def _pil_to_np_mask(path: str) -> np.ndarray:
    try:
        m = Image.open(path).convert("L")
        arr = np.array(m, dtype=np.uint8)  # 0/1/2
        return arr
    except Exception as e:
        raise RuntimeError(f"[GRAPE] Failed to load mask: {path} | {e}")

# ----------------------------- Base Dataset ----------------------------- #

class GrapeDataset(Dataset):
    """
    GRAPE Dataset Base
    - 由 Builder 传入 split 专属 CSV（与 MIMIC 对齐）
    - task_type ∈ {'reid','seg','vf_reg','ue'}
    """
    VF_COLS_DEFAULT: Tuple[str, ...] = tuple([f"vf_{i:02d}" for i in range(1, 62)])

    def __init__(
        self,
        csv_path: str,
        split: Literal["train", "val", "validate", "test"] = "train",
        transform: Optional[Callable[..., Any]] = None,
        task_type: Literal["reid", "seg", "vf_reg", "ue"] = "reid",
        vf_cols: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()
        self.logger = get_logger()
        self.csv_path = csv_path
        self.split = "val" if split == "validate" else split
        self.transform = transform
        self.task_type = task_type
        self.vf_cols = list(vf_cols) if vf_cols is not None else list(self.VF_COLS_DEFAULT)

        df = pd.read_csv(csv_path)
        self.df = df.reset_index(drop=True)

        self.logger.info(f"[GRAPE] Loaded {self.split} split: {len(self.df)} rows, "
              f"{self.df['subject_id'].nunique()} subjects")

        # task-specific setup
        if task_type == "reid":
            self._setup_reid()
        elif task_type == "seg":
            self._setup_seg()
        elif task_type == "vf_reg":
            self._setup_vf()
        elif task_type == "ue":
            self._setup_ue()
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    # -------------------- task setups -------------------- #

    def _setup_reid(self):
        uniq = sorted(self.df["subject_id"].unique())
        self.subject_to_label = {sid: i for i, sid in enumerate(uniq)}
        self.label_to_subject = {i: sid for sid, i in self.subject_to_label.items()}
        self.df["reid_label"] = self.df["subject_id"].map(self.subject_to_label).astype(int)

        self.num_classes = len(uniq)
        cnt = self.df["reid_label"].value_counts()
        self.logger.info(f"[GRAPE:ReID] classes={self.num_classes}, images/subject "
              f"min={int(cnt.min())}, max={int(cnt.max())}, mean={cnt.mean():.2f}")

    def _setup_seg(self):
        # 仅做存在性检查；找不到时尽早抛错（与“数据侧不做过滤”的原则一致）
        if "mask_path" not in self.df.columns:
            raise ValueError("[GRAPE:Seg] CSV missing 'mask_path' column.")
        # 可在这里统计一下 0/1/2 的占比（可选，不打印）

    def _setup_vf(self):
        # 选择存在的 VF 列
        existing = [c for c in self.vf_cols if c in self.df.columns]
        if not existing:
            raise ValueError("[GRAPE:VF] No VF columns found in CSV.")
        self.vf_cols = existing
        self.logger.info(f"[GRAPE:VF] Using VF cols: n={len(self.vf_cols)}")

    def _setup_ue(self):
        """UE 视图：与 MIMIC 类似，提供 reid_label + meta（原样键值）。"""
        uniq = sorted(self.df["subject_id"].unique())
        self.ue_subject_to_label = {sid: i for i, sid in enumerate(uniq)}
        self.df["reid_label"] = self.df["subject_id"].map(self.ue_subject_to_label).astype(int)
        self.num_classes_reid = len(uniq)
        self._has_reid = True

    # -------------------- std API -------------------- #

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, do_transform=True):
        row = self.df.iloc[idx]
        img_np = _pil_to_np_rgb(row["image_path"])

        # 分割任务需要同步处理 mask
        if self.task_type == "seg":
            mask_np = _pil_to_np_mask(row["mask_path"])
            if self.transform and do_transform:
                image, mask_t = self.transform(img_np, mask_np)
            else:
                image = _to_tensor(img_np)                      
                mask_t = torch.as_tensor(mask_np).long()
            return self._get_seg_item(row, image, mask_t)

        # 其他任务：只处理 image
        if self.transform and do_transform:
            image = self.transform(img_np)
        else:
            image = _to_tensor(img_np)

        if self.task_type == "reid":
            return self._get_reid_item(row, image)
        if self.task_type == "vf_reg":
            return self._get_vf_item(row, image)
        if self.task_type == "ue":
            return self._get_ue_item(row, image)
        raise RuntimeError("Unreachable")

    # -------------------- item builders -------------------- #

    def _get_reid_item(self, row, image):
        # 与 MIMIC 对齐：返回 label=reid_label；其余元数据保持原样键
        return {
            "image": image,
            "label": int(row["reid_label"]),
            "subject_id": row["subject_id"],          # 原始键（可为 int/str；来自 CSV）
            "dicom_id": row.get("dicom_id", row.get("image_id", os.path.splitext(os.path.basename(row["image_path"]))[0])),
            "study_id": row["study_id"],              # 你已转换为全局唯一 key
            # MIMIC 里有 view_position；GRAPE 没有则给个占位
            "view_position": row.get("ViewPosition", "NA"),
            "image_path": row["image_path"],
        }

    def _get_seg_item(self, row, image, mask_t: torch.Tensor):
        return {
            "image": image,
            "mask": mask_t,                           # 0/1/2
            "subject_id": row["subject_id"],
            "dicom_id": row.get("dicom_id", row.get("image_id", os.path.splitext(os.path.basename(row["image_path"]))[0])),
            "study_id": row["study_id"],
            "image_path": row["image_path"],
        }

    def _get_vf_item(self, row, image):
        vf_vals = []
        vf_mask = []
        for c in self.vf_cols:
            v = row[c]
            if pd.isna(v):
                vf_vals.append(-1.0)
                vf_mask.append(0.0)
            else:
                vv = float(v)
                vf_vals.append(vv)
                vf_mask.append(0.0 if vv < -0.5 else 1.0)  # 约定：-1 表示缺失
        vf = torch.tensor(vf_vals, dtype=torch.float32)
        vfm = torch.tensor(vf_mask, dtype=torch.float32)
        return {
            "image": image,
            "vf": vf,                 # [61]
            "vf_mask": vfm,           # [61] 0/1
            "subject_id": row["subject_id"],
            "dicom_id": row.get("dicom_id", row.get("image_id", os.path.splitext(os.path.basename(row["image_path"]))[0])),
            "study_id": row["study_id"],
            "image_path": row["image_path"],
        }

    def _get_ue_item(self, row, image):
        # 与 MIMIC 的 UE 结构对齐：targets.reid + meta（保留原始键）
        reid_target = int(row["reid_label"])
        dicom_id = row.get("dicom_id", row.get("image_id", os.path.splitext(os.path.basename(row["image_path"]))[0]))
        return {
            "image": image,
            "targets": {
                "reid": reid_target,
            },
            "meta": {
                "subject_id": row["subject_id"],
                "dicom_id": dicom_id,
                "study_id": row["study_id"],
                "image_path": row["image_path"],
            },
        }

    # -------------------- sampler hook -------------------- #
    def labels_for_sampling(self, kind: str = "reid") -> torch.Tensor:
        assert kind == "reid", "Only 'reid' sampling is supported."
        return torch.as_tensor(self.df["reid_label"].values, dtype=torch.long)


# -------------------------- Task-Specific Subclasses -------------------------- #

class GrapeReIDDataset(GrapeDataset):
    def __init__(self, *args, **kwargs):
        kwargs["task_type"] = "reid"
        super().__init__(*args, **kwargs)

class GrapeSegDataset(GrapeDataset):
    def __init__(self, *args, **kwargs):
        kwargs["task_type"] = "seg"
        super().__init__(*args, **kwargs)

class GrapeVFRegDataset(GrapeDataset):
    def __init__(self, *args, **kwargs):
        kwargs["task_type"] = "vf_reg"
        super().__init__(*args, **kwargs)

class GrapeUEDataset(GrapeDataset):
    def __init__(self, *args, **kwargs):
        kwargs["task_type"] = "ue"
        super().__init__(*args, **kwargs)
