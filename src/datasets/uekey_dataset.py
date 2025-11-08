from __future__ import annotations
from typing import Dict, Any, List
import torch
from torch.utils.data import Dataset, ConcatDataset
from omegaconf import DictConfig
from ..utils.config import get_config, require_config
from ..core.ue_keys import extract_key 

class UEConcatDataset(ConcatDataset):
    """
    ConcatDataset with labels_for_sampling support.
    All child datasets must implement labels_for_sampling(kind), concatenated in order.
    """
    def labels_for_sampling(self, kind: str = "reid") -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for ds in self.datasets:
            if not hasattr(ds, "labels_for_sampling"):
                raise RuntimeError("UEConcatDataset requires child datasets to implement labels_for_sampling(...)")
            lab = ds.labels_for_sampling(kind)
            lab = lab.to(torch.long) if isinstance(lab, torch.Tensor) else torch.as_tensor(lab, dtype=torch.long)
            parts.append(lab)
        if not parts:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(parts, dim=0)


class UEKeyDataset(Dataset):
    """
    Wrap a base dataset and add a 'key' field to each sample.

    key_spec (DictConfig):
      - type: "samplewise" | "classwise"          # for consistency check
      - from: "index" | "field" | "filename"      # the source of key
      - field: 当 from=="field" 时的路径（支持点路径，如 "targets.reid"）
      - lower: bool (默认 True)                    # the string key is lowercase
      - strip: bool (默认 True)                    # the string key is strip
      - namespace: str (可选)                      # only for recording/export, not for concatenation
    """

    def __init__(self, base: Dataset, key_spec: DictConfig):
        if not isinstance(key_spec, DictConfig):
            raise TypeError("UEKeyDataset only accepts DictConfig type for key_spec")
        self.base = base
        self._kspec: DictConfig = key_spec

        self._ktype: str = require_config(self._kspec, "type", type_=str)
        self._kfrom: str = get_config(self._kspec, "from", "field")
        self._ffield: str | None = get_config(self._kspec, "field", None, type_=str)

        self._lower: bool = bool(get_config(self._kspec, "lower", True))
        self._strip: bool = bool(get_config(self._kspec, "strip", True))
        
    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name: str):
        # Forward methods/attributes to the underlying dataset (e.g., labels_for_sampling)
        return getattr(self.base, name)

    def labels_for_sampling(self, kind: str = "reid") -> torch.Tensor:
        if hasattr(self.base, "labels_for_sampling"):
            return self.base.labels_for_sampling(kind)
        raise RuntimeError("Underlying dataset does not implement labels_for_sampling(...)")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.base[idx]
        key = extract_key(s, idx, self._kspec)

        ktype = str(self._ktype)
        if ktype not in ("classwise", "samplewise"):
            raise ValueError(f"Invalid ue.key.type: {ktype}")

        s["key"] = key
        return s
