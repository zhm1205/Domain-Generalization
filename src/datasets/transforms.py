# file: src/utils/transforms_v2.py
from __future__ import annotations
from typing import Any, Callable, List, Optional, Tuple, Literal

import torch
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import InterpolationMode
from torchvision import tv_tensors


# ----------------------------- helpers ----------------------------- #

_to_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])  # -> [0,1]

def _parse_size(image_size: Any) -> Tuple[int, int]:
    """
    Accept (H, W) or (C, H, W) or int. Return (H, W).
    """
    if isinstance(image_size, (list, tuple)):
        if len(image_size) == 2:
            h, w = int(image_size[0]), int(image_size[1])
        elif len(image_size) == 3:
            _, h, w = int(image_size[0]), int(image_size[1]), int(image_size[2])
        else:
            raise ValueError(f"Invalid image_size: {image_size}")
    else:
        h = w = int(image_size)
    return h, w


# ----------------------------- classification / ReID etc. ----------------------------- #

def get_transforms(
    split: Literal["train", "val", "validate", "test"] = "train",
    image_size: Any = (256, 256),
    *,
    normalize: bool = True,
    geom_aug: bool = True,
    pixel_aug: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Callable[[Any], torch.Tensor]:
    """
    torchvision v2 pipeline for single-image tasks.
    Input can be: PIL.Image, numpy(HWC, uint8/float), or torch.Tensor (CHW).
    Output: torch.Tensor (CHW, float32). If normalize=True, it's standardized by mean/std.
    """
    h, w = _parse_size(image_size)

    ops: List[Callable] = [
        # 统一入口：把 PIL / numpy / tensor 转为 tv_tensors.Image
        T.ToImage(),                          # -> tv_tensors.Image
        T.ToDtype(torch.float32, scale=True), # uint8 会被 /255 -> [0,1]
    ]

    if split in ("train",):
        if geom_aug:
            ops += [
                T.RandomResizedCrop(size=(h, w), scale=(0.6, 1.0), ratio=(0.9, 1.1), antialias=True),
                T.RandomAffine(
                    degrees=20,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                T.RandomPerspective(distortion_scale=0.2, p=0.5),
                T.RandomHorizontalFlip(p=0.5),
            ]
        else:
            ops += [T.Resize(size=(h, w), interpolation=InterpolationMode.BILINEAR, antialias=True)]

        if pixel_aug:
            ops += [
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
                T.GaussianBlur(kernel_size=5),      # 近似你的 GaussianBlur
                # Albumentations 的 GaussNoise / CoarseDropout：
                # 这里用 RandomErasing 作为“遮挡”近似；高斯噪声可按需自定义模块再加。
                T.RandomErasing(p=0.2, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
            ]
    else:
        ops += [T.Resize(size=(h, w), interpolation=InterpolationMode.BILINEAR, antialias=True)]

    if normalize:
        ops += [T.Normalize(mean=mean, std=std)]

    # 输出：CHW float32 tensor
    pipeline = T.Compose(ops)

    def _apply(img: Any) -> torch.Tensor:
        # 支持直接传入 numpy / PIL / Tensor
        out = pipeline(img)
        # out 已是 torch.Tensor[CHW, float32]
        return out

    return _apply


# ----------------------------- segmentation (image + mask) ----------------------------- #

def get_seg_transforms(
    split: Literal["train", "val", "validate", "test"] = "train",
    image_size: Any = (256, 256),
    *,
    normalize: bool = True,
    geom_aug: bool = True,
    pixel_aug: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Callable[[Any, Any], Tuple[torch.Tensor, torch.Tensor]]:
    """
    torchvision v2 pipeline for segmentation (image + mask).
    Input:
      - image: PIL / numpy(HWC) / Tensor(CHW)
      - mask : numpy(HW) / Tensor(HW) with integer classes (0/1/2/…)
    Output:
      - image: Tensor(CHW, float32) (normalized if normalize=True)
      - mask : Tensor(HW, long)      (最近邻同步几何变换，不做像素增强/归一化)
    """
    h, w = _parse_size(image_size)

    # 重要：v2 会根据 tv_tensors 类型自动采用合适的插值策略（Mask 用 NEAREST）
    # 我们把 image 包装成 tv_tensors.Image，mask 包装成 tv_tensors.Mask。
    geom_ops: List[Callable] = []
    if split in ("train",) and geom_aug:
        geom_ops += [
            T.RandomResizedCrop(size=(h, w), scale=(0.6, 1.0), ratio=(0.9, 1.1), antialias=True),
            T.RandomAffine(
                degrees=20,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,  # 对 Image 生效；Mask 会自动用最近邻
            ),
            T.RandomHorizontalFlip(p=0.5),
        ]
    else:
        geom_ops += [T.Resize(size=(h, w), interpolation=InterpolationMode.BILINEAR, antialias=True)]

    pixel_ops: List[Callable] = []
    if split in ("train",) and pixel_aug:
        pixel_ops += [
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
            T.GaussianBlur(kernel_size=5),
            # 不对 mask 做任何像素层面的改动
        ]

    norm_ops: List[Callable] = [T.Normalize(mean=mean, std=std)] if normalize else []

    # 把几部分串起来；这些算子在 dict/tv_tensors 上都会共享同一随机参数
    pipeline = T.Compose(
        [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
        + geom_ops
        + pixel_ops
        + norm_ops
    )

    def _apply(image: Any, mask: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        # 包装 mask 为 tv_tensors.Mask（会触发最近邻插值）
        if isinstance(mask, torch.Tensor):
            mask_t = mask
        else:
            mask_t = torch.as_tensor(mask)
        # 确保整型类别
        if not torch.is_floating_point(mask_t):
            mask_t = mask_t.long()
        else:
            mask_t = mask_t.round().long()

        # 使用“样本字典”调用，让 v2 在几何算子上对 image/mask 共享随机参数
        sample = {
            "image": image,                # 可以是 PIL / numpy / Tensor
            "mask": tv_tensors.Mask(mask_t),  # 明确告知其是语义 mask
        }
        out = pipeline(sample)             # v2 会返回等键的 dict

        img_out: torch.Tensor = out["image"]       # CHW float32
        mask_out_tv = out["mask"]                  # tv_tensors.Mask
        mask_out: torch.Tensor = torch.as_tensor(mask_out_tv)  # -> plain Tensor(HW)

        return img_out, mask_out.long()

    return _apply
