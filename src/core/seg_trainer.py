# src/trainers/grape_seg.py
from __future__ import annotations
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trainer_base import TrainerBase
from omegaconf import DictConfig
from ..utils.config import get_config, require_config
from monai.losses import DiceCELoss

class SegTrainer(TrainerBase):
    """
    Supervised trainer for OD/OC segmentation (3 classes: 0 bg, 1 OD, 2 OC).
    - Batch must provide: image, mask (Long[B,H,W] with values in {0,1,2})
    - Model forward returns: feats, logits where logits is [B,3,H,W]
    - Loss: MONAI DiceCELoss (dice + CE in one). Configurable.
    """

    def __init__(self, config: DictConfig, device: torch.device, evaluation_strategy):
        super().__init__(config, device)
        self.evaluation_strategy = evaluation_strategy

        # Loss config (mirrors MONAI DiceCELoss signature subset)
        crit_cfg = get_config(config, "training.criterion", DictConfig({}))
        self.include_background = bool(get_config(crit_cfg, "include_background", False))
        self.squared_pred = bool(get_config(crit_cfg, "squared_pred", False))
        self.jaccard = bool(get_config(crit_cfg, "jaccard", False))
        self.lambda_dice = float(get_config(crit_cfg, "lambda_dice", 1.0))
        self.lambda_ce = float(get_config(crit_cfg, "lambda_ce", 1.0))
        self._loss = self._build_loss()

        self.main_metric = get_config(self.config, "evaluation.main_metric", "dice_avg")
        self._best_value = float("-inf")
        self.best_metrics = {}   # 若项目里已有，可忽略

    def _build_loss(self) -> nn.Module:
        return DiceCELoss(
            include_background=self.include_background,
            to_onehot_y=False,            # y is already class-indexed tensor [B,H,W]
            sigmoid=True,
            squared_pred=self.squared_pred,
            jaccard=self.jaccard,
            lambda_dice=self.lambda_dice,
            lambda_ce=self.lambda_ce,
            reduction="mean",
        )
        
    def _init_epoch_metrics(self) -> Dict[str, Any]:
        """Initialize metrics for supervised training"""
        from ..utils.metrics import AverageMeter
        return {
            'loss': AverageMeter()
        }

    # --- 在 SegTrainer 类里新增/覆盖 ---
    def _is_best_model(self, eval_stats: Dict[str, float]) -> bool:
        """
        用 main_metric（默认 dice_avg）判定是否为 best。
        """
        if eval_stats is None:
            return False
        val = eval_stats.get(self.main_metric, None)
        if val is None:
            # 没有该指标就不触发 best
            return False
        if val > self._best_value:
            self._best_value = val
            # 记录一份当前最优指标，CheckpointHook 里通常也会写入
            self.best_metrics = dict(eval_stats)
            return True
        return False

    def run_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.optimizer.zero_grad()
        x = batch["image"].to(self.device)          # [B,3,H,W]
        mask = batch["mask"].to(self.device).long()    # [B,H,W] with {0,1,2}
        cup_mask  = (mask == 2).float()              # [B,H,W]
        disc_mask = (mask > 0).float()               # [B,H,W]（盘包含杯）
        target    = torch.stack([cup_mask, disc_mask], dim=1)   # [B,2,H,W]

        logits = self.model(x)                   # [B,3,H,W]
        loss = self._loss(logits, target)

        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.item())}
