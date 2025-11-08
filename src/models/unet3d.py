# -*- coding: utf-8 -*-
# src/models/unet3d.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.registry import register_model
from src.utils.config import get_config
from src.utils.logger import get_logger


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm="instance", act="relu", dropout=0.0):
        super().__init__()
        Norm = nn.InstanceNorm3d if str(norm).lower().startswith("instance") else nn.BatchNorm3d
        Act  = nn.ReLU if str(act).upper()=="RELU" else nn.LeakyReLU
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch, affine=True),
            Act(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch, affine=True),
            Act(inplace=True),
            nn.Dropout3d(p=float(dropout)) if dropout and dropout>0 else nn.Identity(),
        )
    def forward(self, x): return self.block(x)


class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch, **kw):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock3D(in_ch, out_ch, **kw)
    def forward(self, x): return self.conv(self.pool(x))


class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, **kw):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
            self.reduce = nn.Conv3d(in_ch, in_ch//2, 1, bias=False)
            mid = in_ch//2
        else:
            self.up = nn.ConvTranspose3d(in_ch, in_ch//2, 2, stride=2)
            self.reduce = nn.Identity()
            mid = in_ch//2
        self.conv = ConvBlock3D(mid + out_ch, out_ch, **kw)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce(x)
        # 尺寸对齐
        dz = skip.shape[-3] - x.shape[-3]
        dy = skip.shape[-2] - x.shape[-2]
        dx = skip.shape[-1] - x.shape[-1]
        if dz or dy or dx:
            x = nn.functional.pad(x, (0, max(dx,0), 0, max(dy,0), 0, max(dz,0)))
            x = x[..., :skip.shape[-3], :skip.shape[-2], :skip.shape[-1]]
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


@register_model("unet3d")
class UNet3D(nn.Module):
    """返回 (feats, logits) 以兼容 SegTrainer。"""
    def __init__(self, cfg: DictConfig | Dict[str, Any], in_channels: Optional[int]=None, eps: Optional[float]=None):
        super().__init__()
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        self.logger = get_logger()

        c_in_cfg = get_config(cfg, "in_channels", 4)
        self.in_channels = in_channels if in_channels is not None else int(c_in_cfg)
        self.num_classes = int(get_config(cfg, "out_channels", get_config(cfg, "num_classes", 2)))

        base = int(get_config(cfg, "base_channels", 16))
        chs: List[int] = list(get_config(cfg, "channels", [base, 32, 64, 128, 256]))
        assert len(chs) >= 4
        bilinear = bool(get_config(cfg, "up_bilinear", True))
        norm = get_config(cfg, "norm", "instance")
        act  = get_config(cfg, "act", "relu")
        dropout = float(get_config(cfg, "dropout", 0.0))

        self.enc0 = ConvBlock3D(self.in_channels, chs[0], norm=norm, act=act, dropout=dropout)
        self.down1 = Down3D(chs[0], chs[1], norm=norm, act=act, dropout=dropout)
        self.down2 = Down3D(chs[1], chs[2], norm=norm, act=act, dropout=dropout)
        self.down3 = Down3D(chs[2], chs[3], norm=norm, act=act, dropout=dropout)
        bott_ch = chs[4] if len(chs)>4 else chs[-1]*2
        self.bottleneck = Down3D(chs[3], bott_ch, norm=norm, act=act, dropout=dropout)

        self.up3 = Up3D(bott_ch, chs[3], bilinear=bilinear, norm=norm, act=act, dropout=dropout)
        self.up2 = Up3D(chs[3], chs[2], bilinear=bilinear, norm=norm, act=act, dropout=dropout)
        self.up1 = Up3D(chs[2], chs[1], bilinear=bilinear, norm=norm, act=act, dropout=dropout)
        self.up0 = Up3D(chs[1], chs[0], bilinear=bilinear, norm=norm, act=act, dropout=dropout)

        self.head = nn.Conv3d(chs[0], self.num_classes, 1)

        self.logger.info(f"[UNet3D] in={self.in_channels}, classes={self.num_classes}, chs={chs}, bilinear={bilinear}")

    def forward(self, x: torch.Tensor):
        e0 = self.enc0(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        b  = self.bottleneck(e3)

        d3 = self.up3(b,  e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        d0 = self.up0(d1, e0)

        logits = self.head(d0)
        feats = {"enc": [e0, e1, e2, e3], "dec": [d3, d2, d1, d0], "bottleneck": b}
        return feats, logits
