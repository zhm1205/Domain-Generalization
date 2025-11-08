# src/evaluation/grape_eval.py
from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from ..registry import register_evaluation_strategy
from ..utils.config import get_config

from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.losses import DiceCELoss

# ------------------------- VF Regression Evaluation ------------------------- #

@register_evaluation_strategy("grape_vf_reg")
class GrapeVFRegressionEvaluationStrategy:
    """
    Evaluation for 61-dim masked VF regression.
    Metrics:
      - loss (same as training; masked MSE or Huber)
      - mae, rmse (masked, aggregated across dataset)
      - optional: mae_per_point (disabled by default)
    """

    def __init__(self, config: DictConfig):
        self.config = config or {}
        crit_cfg = get_config(config, "training.criterion", {})
        self.kind = str(get_config(crit_cfg, "name", "mse")).lower()  # 'mse' | 'huber'
        self.delta = float(get_config(crit_cfg, "delta", 1.0))
        ev_cfg = get_config(config, "evaluation", {})
        self.return_per_point = bool(get_config(ev_cfg, "return_per_point", False))

    @torch.no_grad()
    def evaluate_epoch(self, model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict[str, float]:
        model.eval()
        total_loss_num = 0.0
        total_loss_den = 0.0
        mae_num = 0.0
        mse_num = 0.0
        den = 0.0

        # optional per-point
        pp_mae_num = None
        pp_mse_num = None
        pp_den = None

        pbar = tqdm(data_loader, desc="Evaluate VF-REG", leave=False)
        for batch in pbar:
            x = batch["image"].to(device)
            y = batch["vf"].to(device).float()
            m = batch["vf_mask"].to(device).float()

            _, preds = model(x)  # [B,61]

            # loss (masked)
            if self.kind == "mse":
                loss_el = (preds - y) ** 2
            else:
                loss_el = F.smooth_l1_loss(preds, y, reduction="none", beta=self.delta)
            num = (loss_el * m).sum().item()
            d = m.sum().item()
            total_loss_num += num
            total_loss_den += max(d, 1e-8)

            # metrics: mae, rmse
            abs_err = (preds - y).abs()
            mae_num += (abs_err * m).sum().item()
            mse_num += (loss_el if self.kind == "mse" else (preds - y) ** 2).mul(m).sum().item()
            den += d

            if self.return_per_point:
                B, K = preds.shape
                if pp_mae_num is None:
                    pp_mae_num = torch.zeros(K, dtype=torch.float64, device=device)
                    pp_mse_num = torch.zeros(K, dtype=torch.float64, device=device)
                    pp_den = torch.zeros(K, dtype=torch.float64, device=device)
                pp_mae_num += (abs_err * m).sum(dim=0).double()
                pp_mse_num += (((preds - y) ** 2) * m).sum(dim=0).double()
                pp_den += m.sum(dim=0).double()

        metrics = {
            "loss": float(total_loss_num / max(total_loss_den, 1e-8)),
            "mae": float(mae_num / max(den, 1e-8)),
            "rmse": float(np.sqrt(mse_num / max(den, 1e-8))),
        }

        if self.return_per_point and pp_den is not None:
            pp_mae = (pp_mae_num / (pp_den.clamp_min(1e-8))).cpu().numpy()
            for i, v in enumerate(pp_mae.tolist(), start=1):
                metrics[f"mae@{i:02d}"] = float(v)
        return metrics


# ------------------------- Segmentation Evaluation ------------------------- #

def _ensure_same_hw(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Resize logits to match y spatial size if necessary."""
    _, _, H, W = logits.shape
    Hy, Wy = y.shape[-2], y.shape[-1]
    if (H, W) != (Hy, Wy):
        logits = F.interpolate(logits, size=(Hy, Wy), mode="bilinear", align_corners=False)
    return logits


def _confusion_per_class(pred: torch.Tensor, gt: torch.Tensor, num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute TP, FP, FN for each class on flattened tensors.
    pred/gt: [N] long with {0..C-1}
    """
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        p = (pred == c)
        g = (gt == c)
        tp[c] = int((p & g).sum())
        fp[c] = int((p & ~g).sum())
        fn[c] = int((~p & g).sum())
    return tp, fp, fn


@register_evaluation_strategy("grape_seg")
class GrapeSegmentationEvaluationStrategy:
    """
    Evaluation for GRAPE OD/OC segmentation using MONAI metrics.

    Assumptions:
      - Dataset returns:
          batch["image"] -> FloatTensor [B,3,H,W]
          batch["mask"]  -> LongTensor  [B,H,W] with {0:bg, 1:disc, 2:cup}
      - We evaluate in two-channel binary space: [cup, disc]
        GT: cup = (mask==2), disc = (mask>0)

    Config keys (optional):
      evaluation.seg:
        threshold: float                         (default: 0.75)
        project_cup_inside_disc: bool            (default: True, only for sigmoid2)
        class_indices:                           (for softmax3; default: bg=0,disc=1,cup=2)
          bg: 0
          disc: 1
          cup: 2
      training.criterion (for reporting loss, optional):
        include_background: False
        softmax: False           # with sigmoid2
        squared_pred: False
        jaccard: False
        lambda_dice: 1.0
        lambda_ce: 1.0
        ce_weight: null
    """

    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or DictConfig({})

        seg_cfg = get_config(self.config, "evaluation.seg", DictConfig({}))
        self.threshold: float = float(get_config(seg_cfg, "threshold", 0.5))
        self.project_cup_inside_disc: bool = bool(get_config(seg_cfg, "project_cup_inside_disc", True))

        # softmax3 class index mapping
        ci = get_config(seg_cfg, "class_indices", DictConfig({}))
        self.idx_bg   = int(get_config(ci, "bg",   0))
        self.idx_disc = int(get_config(ci, "disc", 1))
        self.idx_cup  = int(get_config(ci, "cup",  2))

        # MONAI metrics
        # two foreground channels => include_background=False
        self.dice_metric = DiceMetric(include_background=True, reduction="none")  # -> [2], per-class mean over batch
        self.miou_metric = MeanIoU(include_background=True, reduction="mean")          # -> scalar

        # Optional loss for reporting (align with training if desired)
        crit_cfg = get_config(self.config, "training.criterion", DictConfig({}))
        include_background = bool(get_config(crit_cfg, "include_background", True))
        squared_pred = bool(get_config(crit_cfg, "squared_pred", False))
        jaccard = bool(get_config(crit_cfg, "jaccard", False))
        lambda_dice = float(get_config(crit_cfg, "lambda_dice", 1.0))
        lambda_ce = float(get_config(crit_cfg, "lambda_ce", 1.0))

        # Note: for two-channel multilabel, set softmax=False, to_onehot_y=False, include_background=False
        self.loss_fn = DiceCELoss(
            include_background=include_background,
            sigmoid=True,
            squared_pred=squared_pred,
            jaccard=jaccard,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            reduction="mean",
        )

    @torch.no_grad()
    def evaluate_epoch(self, model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict[str, float]:
        model.eval()
        model.to(device)

        total_loss = 0.0
        n_samples = 0

        # reset accumulators
        self.dice_metric.reset()
        self.miou_metric.reset()

        pbar = tqdm(data_loader, desc="Evaluate SEG (GRAPE)", leave=False)
        for batch in pbar:
            x = batch["image"].to(device)           # [B,3,H,W]
            y_id = batch["mask"].to(device).long()  # [B,H,W] with {0:bg,1:disc,2:cup}

            # --- build two-channel GT: [cup, disc] ---
            y_cup  = y_id.eq(2).float()
            y_disc = y_id.gt(0).float()
            y_bin  = torch.stack([y_cup, y_disc], dim=1)  # [B,2,H,W]

            # --- forward ---
            logits = model(x)
            pred_bin = torch.sigmoid(logits).ge(self.threshold)

            # --- accumulate MONAI metrics ---
            self.dice_metric(y_pred=pred_bin, y=y_bin)
            self.miou_metric(y_pred=pred_bin, y=y_bin)
            # For DiceCELoss with multilabel two-channel targets, pass logits + y_bin
            loss = self.loss_fn(logits, y_bin)
            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            n_samples += bs

        # ---- aggregate ----
        dc = self.dice_metric.aggregate().mean(axis=0)
        cup_dc = dc[0].item()
        disc_dc = dc[1].item()
        avg_dc = dc.mean().item()

        miou = float(self.miou_metric.aggregate().item())  # scalar

        metrics = {
            "loss": float(total_loss / max(1, n_samples)),
            "cup_dc":  cup_dc,
            "disc_dc": disc_dc,
            "avg_dc":  avg_dc,
            "miou":    miou,   # == jc
            "jc":      miou,   # alias, if you prefer this key
        }

        # reset for next epoch call
        self.dice_metric.reset()
        self.miou_metric.reset()

        return metrics
