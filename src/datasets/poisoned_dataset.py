from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict

import torch
from torch.utils.data import Dataset

from ..core.ue_artifacts import UEShardsAccessor
from ..core.ue_keys import extract_key
from albumentations.pytorch import ToTensorV2

def _normalize_inplace(img: torch.Tensor, mean, std):
    """
    In-place per-channel normalize. `img` is float tensor in [0,1], shape [C,H,W].
    """
    assert img.ndim == 3, f"Expected 3D tensor [C,H,W], got {img.shape}"
    for c, (m, s) in enumerate(zip(mean, std)):
        img[c].sub_(float(m)).div_(float(s))
    return img


class _LRU:
    """Tiny LRU for objects like memmapped arrays or per-class noise tensors."""
    def __init__(self, max_items: int = 16):
        self.max = int(max_items)
        self.od = OrderedDict()
    def get(self, key, loader):
        if key in self.od:
            v = self.od.pop(key); self.od[key] = v; return v
        v = loader(); self.od[key] = v
        if len(self.od) > self.max: self.od.popitem(last=False)
        return v


class PoisonedDataset(Dataset):
    """
    Wrap a base dataset and inject perturbations on-the-fly (train split only).
    Injection happens BEFORE Normalize; we then normalize inside this wrapper.

    NOTE:
    - Keys can be any JSON-safe raw key (int/str/list/tuple->list/...), and will be
      passed as-is to the artifact accessor/provider.  # [INFO]
    """

    def __init__(
        self,
        base: Dataset,
        *,
        perturb_type: str,                 # "classwise" | "samplewise"
        key_spec: Dict[str, Any],          # {"type","from","field"}
        source_cfg: Dict[str, Any],        # {"type":"shards"|"provider", ...}
        clamp: Tuple[float, float] = (0.0, 1.0),
        apply_stage: str = "before_normalize",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        io_cache_max_items: int = 16,
        provider_instance: Any = None,     # allow passing a prebuilt provider
    ):
        self.base = base
        self.perturb_type = perturb_type
        self.key_spec = key_spec
        self.clamp_min, self.clamp_max = map(float, clamp)
        self.apply_stage = apply_stage
        self.mean = tuple(mean); self.std = tuple(std)
        self._cache = _LRU(io_cache_max_items)

        # ---------------- source wiring ----------------
        if provider_instance is not None:
            self.provider = provider_instance
            self._get_noise = self._get_noise_from_provider
            self.accessor = None
        else:
            stype = str(source_cfg.get("type", "shards")).lower()
            if stype == "shards":
                self.accessor = UEShardsAccessor.from_manifest(source_cfg["manifest_path"])
                self._get_noise = self._get_noise_from_shards
                self.provider = None
            elif stype == "provider":
                raise RuntimeError(
                    "[UE] source.type='provider' but provider_instance is not provided；"
                    "please build and pass the provider via build_unlearnable_provider_instance(...) and attach_unlearnable_noise(...) outside."
                )
            else:
                raise ValueError(f"Unknown source.type: {stype}")

        if self.apply_stage != "before_normalize":
            raise ValueError("PoisonedDataset requires apply_stage='before_normalize'.")

    # ---------- delegation so samplers/others can see base attrs ----------
    def __getattr__(self, name: str):
        """
        Delegate unknown attributes to the base dataset.
        Called only if normal attribute lookup fails on self.
        """
        return getattr(self.base, name)

    def labels_for_sampling(self, kind: str = "reid") -> torch.Tensor:
        """
        Forward labels to the sampler. Required by PKPolicy / MPerClassSampler.
        """
        if hasattr(self.base, "labels_for_sampling"):
            return self.base.labels_for_sampling(kind)
        raise RuntimeError("Base dataset does not implement labels_for_sampling(...)")

    # ---------- dataset protocol ----------
    def __len__(self): 
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base.__getitem__(idx, do_transform=False)

        img: torch.Tensor = sample["image"].float()

        ktype, key = self.key_spec["type"], extract_key(sample, idx, self.key_spec)
        if str(ktype) != self.perturb_type:
            raise ValueError(f"perturb_type mismatch: wrapper={self.perturb_type}, key={ktype}")

        noise = self._get_noise(key)   # raw key, no int() cast 
        C_noise, H_noise, W_noise = noise.shape
        C_image, H_image, W_image = img.shape
        
        if C_noise == 1 and C_image > 1:
            noise = noise.repeat(C_image, 1, 1)
        
        if H_noise != H_image or W_noise != W_image:
            raise ValueError(f"noise shape mismatch: noise={noise.shape}, img={img.shape}")

        if noise.device != img.device:
            noise = noise.to(img.device)
        img = torch.clamp(img + noise, min=self.clamp_min, max=self.clamp_max)

        # do transform after adding noise
        if getattr(self.base, "task_type", None) == "seg":
            if self.transform is None:
                raise RuntimeError("transform is required for seg")
            mask = sample["mask"]                     # 确保 base 在 do_transform=False 时也返回原始 mask（HW, long/uint）
            img, mask = self.transform(img, mask)     # v2 风格：返回两个 tensor
            sample["mask"] = mask.long()
        else:
            if self.transform is None:
                raise RuntimeError("transform is required")
            img = self.transform(img)                 # v2 风格：直接返回 tensor

        if self.apply_stage == "before_normalize":
            _normalize_inplace(img, self.mean, self.std)

        sample["image"] = img
        return sample

    # ---------- noise resolvers ----------
    def _get_noise_from_shards(self, key) -> torch.Tensor:
        """
        Fetch noise via artifact accessor.
        We pass the raw key as-is, and provide `scope=self.perturb_type` for consistency check.  
        """
        return self.accessor.get(key, perturb_type=self.perturb_type)  

    def _get_noise_from_provider(self, key) -> torch.Tensor:
        """
        Fetch noise via provider instance and cache per key.
        """
        def _load():
            return self.provider.get_noise(key, self.perturb_type)
        return self._cache.get(key, _load)
