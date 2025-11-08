# -*- coding: utf-8 -*-
# src/evaluation/seg_eval_dice.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import numpy as np

from src.registry import register_evaluation_strategy
from src.utils.logger import get_logger

def _resize_like(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    将 logits 的空间大小对齐 target。
    支持 2D: [B,C,H,W]  和 3D: [B,C,D,H,W]。
    """
    if logits.shape[2:] == target.shape[-len(logits.shape)+2:]:
        return logits
    if logits.ndim == 4:
        _, _, H, W = logits.shape
        Hy, Wy = target.shape[-2], target.shape[-1]
        return F.interpolate(logits, size=(Hy, Wy), mode="bilinear", align_corners=False)
    elif logits.ndim == 5:
        _, _, D, H, W = logits.shape
        Dy, Hy, Wy = target.shape[-3], target.shape[-2], target.shape[-1]
        return F.interpolate(logits, size=(Dy, Hy, Wy), mode="trilinear", align_corners=False)
    else:
        raise ValueError(f"Unsupported logits ndim: {logits.ndim}")

def _dice_per_class(pred: torch.Tensor, gt: torch.Tensor, num_classes: int, eps: float = 1e-6) -> np.ndarray:
    """
    pred/gt 为 Long 类型，形状 [N,...]，取值 {0..C-1}
    返回 C 维 dice 向量（前景/多类平均时可去掉第0类）。
    """
    dices = []
    for c in range(num_classes):
        pc = (pred == c)
        gc = (gt == c)
        inter = (pc & gc).sum().item()
        denom = pc.sum().item() + gc.sum().item()
        d = (2.0 * inter + eps) / (denom + eps)
        dices.append(d)
    return np.array(dices, dtype=np.float32)

@register_evaluation_strategy("seg_dice")
class SegDiceEvaluator:
    """
    统一的分割评估：Dice（支持 2D/3D，多类），自动跳过无标注样本(mask=None)。
    约定：
      - model(x) 返回 (feats, logits)
      - batch: {"image": [B,C,(D,)H,W], "mask": [B,(D,)H,W] or None, "meta": {...}}
    """
    def __init__(self, config: DictConfig):
        self.cfg = config
        self.logger = get_logger()
        self.num_classes: int = int(self.cfg.get("model", {}).get("num_classes", 2))
        # 是否把背景计入平均
        self.include_background: bool = bool(
            self.cfg.get("evaluation", {}).get("include_background", False)
        )

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, data_loader) -> Dict[str, float]:
        model.eval()
        n_with_gt = 0
        dice_sum = np.zeros(self.num_classes, dtype=np.float64)

        total_loss = 0.0  # 如果外层会提供 loss，可填充；此处保留字段
        for batch in data_loader:
            x = batch["image"].to(next(model.parameters()).device, non_blocking=True)
            mask = batch.get("mask", None)
            # 无标注的样本跳过指标计算
            if mask is None or (isinstance(mask, torch.Tensor) and mask.numel() == 0):
                _ = model(x)  # 前向以便兼容需要forward的Hook
                continue

            y = mask.to(x.device, non_blocking=True).long()
            _, logits = model(x)  # [B,C,(D,)H,W]

            # 对齐空间维度
            logits = _resize_like(logits, y)

            # 取 argmax 得到预测
            pred = torch.argmax(logits, dim=1)  # [B,(D,)H,W]

            # 逐 batch 聚合
            dices = _dice_per_class(pred, y, self.num_classes)  # (C,)
            dice_sum += dices
            n_with_gt += 1

        if n_with_gt == 0:
            self.logger.warning("[SegDiceEvaluator] No labeled samples in dataloader; returning empty metrics.")
            return {"dice_avg": 0.0}

        dice_mean = (dice_sum / n_with_gt).astype(np.float32)  # per-class
        if self.include_background:
            dice_avg = float(dice_mean.mean())
        else:
            if self.num_classes > 1:
                dice_avg = float(dice_mean[1:].mean())
            else:
                dice_avg = float(dice_mean.mean())

        # 导出以便日志与早停
        out = {"dice_avg": dice_avg}
        # 同时给出每类 dice
        for c, v in enumerate(dice_mean.tolist()):
            out[f"dice_c{c}"] = float(v)
        return out
