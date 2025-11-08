# -*- coding: utf-8 -*-
# src/datasets/brats_gli_3d.py
from __future__ import annotations
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.registry import register_dataset
from src.utils.config import get_config
from src.utils.logger import get_logger


MODALITIES = ["t1c", "t1n", "t2f", "t2w"]   # 你的示例四模态
SEG_SUFFIX = "seg"


def _load_nii(p: Path) -> np.ndarray:
    arr = nib.load(str(p)).get_fdata(dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _zscore(x: np.ndarray, nonzero: Optional[np.ndarray] = None) -> np.ndarray:
    if nonzero is None:
        nonzero = x != 0
    if nonzero.sum() == 0:
        return x
    mu = x[nonzero].mean()
    sd = x[nonzero].std()
    sd = sd if sd > 1e-6 else 1.0
    return (x - mu) / sd


def _center_crop(vol: np.ndarray, size: Tuple[int,int,int]) -> np.ndarray:
    """vol: [C,D,H,W] 或 [D,H,W]"""
    if vol.ndim == 4:
        C, D, H, W = vol.shape
    else:
        D, H, W = vol.shape
    cd, ch, cw = size
    sd = max((D - cd)//2, 0); sh = max((H - ch)//2, 0); sw = max((W - cw)//2, 0)
    if vol.ndim == 4:
        return vol[:, sd:sd+cd, sh:sh+ch, sw:sw+cw]
    else:
        return vol[sd:sd+cd, sh:sh+ch, sw:sw+cw]


def _start_for_random_fg(mask: Optional[np.ndarray], size: Tuple[int,int,int], shp: Tuple[int,int,int]):
    cd, ch, cw = size
    D, H, W = shp
    Dm, Hm, Wm = max(D-cd,0), max(H-ch,0), max(W-cw,0)
    if mask is not None and mask.any():
        z, y, x = (np.argwhere(mask>0)[np.random.randint(0, mask.sum())]).tolist()
        sd = int(np.clip(z - cd//2, 0, Dm))
        sh = int(np.clip(y - ch//2, 0, Hm))
        sw = int(np.clip(x - cw//2, 0, Wm))
    else:
        sd = 0 if Dm==0 else np.random.randint(0, Dm+1)
        sh = 0 if Hm==0 else np.random.randint(0, Hm+1)
        sw = 0 if Wm==0 else np.random.randint(0, Wm+1)
    return sd, sh, sw


def _crop_at(vol: np.ndarray, start: Tuple[int,int,int], size: Tuple[int,int,int]) -> np.ndarray:
    sd, sh, sw = start; cd, ch, cw = size
    if vol.ndim == 4:
        return vol[:, sd:sd+cd, sh:sh+ch, sw:sw+cw]
    else:
        return vol[sd:sd+cd, sh:sh+ch, sw:sw+cw]


def _pad_to_min(vol: np.ndarray, size: Tuple[int,int,int]) -> np.ndarray:
    """如果体素比 patch 小则对称 pad。"""
    if vol.ndim == 4:
        C, D, H, W = vol.shape
    else:
        D, H, W = vol.shape
    cd, ch, cw = size
    pd = max(cd - D, 0); ph = max(ch - H, 0); pw = max(cw - W, 0)
    if pd==ph==pw==0:
        return vol
    pad_t = (pw//2, pw - pw//2, ph//2, ph - ph//2, pd//2, pd - pd//2)  # (W_left,W_right,H_top,H_bot,D_front,D_back)
    if vol.ndim == 4:
        zeros = ((0,0), (pad_t[4], pad_t[5]), (pad_t[2], pad_t[3]), (pad_t[0], pad_t[1]))
    else:
        zeros = ((pad_t[4], pad_t[5]), (pad_t[2], pad_t[3]), (pad_t[0], pad_t[1]))
    return np.pad(vol, zeros, mode="constant", constant_values=0)


def _collect_paths(case_dir: Path) -> Dict[str, Optional[Path]]:
    paths = {m: None for m in MODALITIES}
    paths["seg"] = None
    for f in case_dir.glob("*.nii.gz"):
        name = f.name.lower()
        for m in MODALITIES:
            if f"-{m}." in name:
                paths[m] = f
        if "-seg." in name:
            paths["seg"] = f
    return {"dir": case_dir, **paths}


@register_dataset("process3d")
class BraTSGLI3DDataset(Dataset):
    """
    输出：
      image: FloatTensor [C,D,H,W]
      mask:  LongTensor  [D,H,W] 或 None（无标注）
      meta:  dict: {'case_id': ...}
    """
    def __init__(self, cfg, split: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.logger = get_logger()

        # 根目录：优先 split 指定，否则 dataset.root
        if split == "train":
            root = get_config(cfg, "dataset.train.root", get_config(cfg, "dataset.root"))
        elif split in ("val", "valid", "validation"):
            root = get_config(cfg, "dataset.val.root", get_config(cfg, "dataset.root"))
        else:  # test
            root = get_config(cfg, "dataset.test.root", get_config(cfg, "dataset.root"))
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"[BraTSGLI3DDataset] root not found: {self.root}")

        # 采样与大小
        self.patch_size = tuple(get_config(cfg, "dataset.patch_size", [96,96,96]))
        assert len(self.patch_size) == 3
        self.random_crop = bool(get_config(cfg, f"dataset.{split}.random_crop",
                                   True if split=="train" else False))
        # 训练默认必须有 seg；测试/验证默认可无 seg
        self.require_seg = bool(get_config(cfg, f"dataset.{split}.require_seg",
                                  True if split=="train" else False))

        # 枚举病例
        self.entries: List[Dict[str, Optional[Path]]] = []
        for case_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            paths = _collect_paths(case_dir)
            if any(paths[m] is None for m in MODALITIES):
                # 若允许缺模态可在这里放行
                continue
            if self.require_seg and (paths["seg"] is None):
                continue
            self.entries.append(paths)

        if len(self.entries) == 0:
            raise RuntimeError(f"[BraTSGLI3DDataset] no valid cases under {self.root}")

        self.logger.info(
            f"[BraTSGLI3DDataset] split={split} root={self.root} cases={len(self.entries)} "
            f"patch={self.patch_size} random_crop={self.random_crop} require_seg={self.require_seg}"
        )

    def __len__(self): return len(self.entries)

    def _load_case(self, idx: int):
        e = self.entries[idx]; case_id = e["dir"].name
        imgs = [ _load_nii(e[m]) for m in MODALITIES ]  # C*[D,H,W]
        vol = np.stack(imgs, axis=0)                    # [C,D,H,W]
        # 逐模态 z-score（在非零体素）
        for c in range(vol.shape[0]):
            vol[c] = _zscore(vol[c], vol[c] != 0)
        seg = None
        if e["seg"] is not None and e["seg"].exists():
            seg = _load_nii(e["seg"]).astype(np.int64)
        return vol, seg, case_id

    def __getitem__(self, idx: int):
        vol, seg, case_id = self._load_case(idx)  # [C,D,H,W], [D,H,W] or None

        # 不足 patch 时 pad
        vol = _pad_to_min(vol, self.patch_size)
        if seg is not None:
            seg = _pad_to_min(seg, self.patch_size)

        C, D, H, W = vol.shape
        cd, ch, cw = self.patch_size

        if self.random_crop:
            start = _start_for_random_fg(seg, self.patch_size, (D,H,W))
            vol = _crop_at(vol, start, self.patch_size)
            if seg is not None:
                seg = _crop_at(seg, start, self.patch_size)
        else:
            vol = _center_crop(vol, self.patch_size)
            if seg is not None:
                seg = _center_crop(seg, self.patch_size)

        out = {
            "image": torch.from_numpy(vol.copy()).float(),            # [C,D,H,W]
            "mask": None if seg is None else torch.from_numpy(seg.copy()).long(),  # [D,H,W]
            "meta": {"case_id": case_id},
        }
        return out


# ------------------------- Builders ------------------------- #

def build_brats_gli_3d_loader(cfg, split: str, shuffle: bool = True) -> DataLoader:
    ds = BraTSGLI3DDataset(cfg, split=split)
    bs = int(get_config(cfg, f"dataset.{split}.batch_size", get_config(cfg, "dataset.batch_size", 1)))
    nw = int(get_config(cfg, f"dataset.{split}.num_workers", get_config(cfg, "dataset.num_workers", 4)))
    pin = bool(get_config(cfg, f"dataset.{split}.pin_memory", True))
    ddp = bool(get_config(cfg, "distributed", False))
    return DataLoader(ds, batch_size=bs, shuffle=(shuffle and not ddp),
                      num_workers=nw, pin_memory=pin, drop_last=False)

def build_train_loader(cfg) -> DataLoader:
    return build_brats_gli_3d_loader(cfg, "train", shuffle=True)

def build_val_loader(cfg) -> DataLoader:
    # 如果你想把 training 再划一部分做 val，可改 dataset.val.root
    return build_brats_gli_3d_loader(cfg, "val", shuffle=False)

def build_test_loader(cfg) -> DataLoader:
    return build_brats_gli_3d_loader(cfg, "test", shuffle=False)
