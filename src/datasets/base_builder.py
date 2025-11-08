# file: src/datasets/base_builder.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Tuple
from abc import ABC, abstractmethod
import warnings, random
from ..utils.logger import get_logger
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, ConcatDataset
from omegaconf import DictConfig, OmegaConf
from ..utils.config import get_config, require_config
from ..registry import get_dataset_builder
from .uekey_dataset import UEConcatDataset

class SamplerPolicy:
    def build(self, dataset: Dataset, split: str) -> Tuple[Optional[torch.utils.data.Sampler], Optional[int]]:
        """Return (sampler, batch_size_override)."""
        raise NotImplementedError

class RandomPolicy(SamplerPolicy):
    def __init__(self, batch_size: int): self.bs = batch_size
    def build(self, dataset, split):
        return (RandomSampler(dataset), self.bs) if split == "train" else (SequentialSampler(dataset), None)

def _make_pk_policy(p: int, k: int) -> SamplerPolicy:
    from pytorch_metric_learning.samplers import MPerClassSampler  # lazy import
    class PKPolicy(SamplerPolicy):
        def build(self, dataset, split):
            if split != "train":
                return SequentialSampler(dataset), None
            if not hasattr(dataset, "labels_for_sampling"):
                raise RuntimeError("PKPolicy requires dataset.labels_for_sampling('reid')")

            labels = dataset.labels_for_sampling("reid")
            # ensure torch.LongTensor, and count unique labels
            if isinstance(labels, torch.Tensor):
                labels = labels.to(torch.long)
                n_labels = int(labels.unique().numel())
            else:
                try:
                    import numpy as np
                    n_labels = int(np.unique(labels).size)
                except Exception:
                    # fallback: convert to tensor then unique
                    labels = torch.as_tensor(labels, dtype=torch.long)
                    n_labels = int(labels.unique().numel())

            # --- key fix: cap P by number of unique labels ---
            eff_p = max(1, min(int(p), n_labels))
            eff_k = int(k)
            bs = eff_p * eff_k

            # also ensure list_size >= batch_size (another MPerClassSampler assert)
            list_size = max(len(dataset), bs)

            sampler = MPerClassSampler(
                labels,
                m=eff_k,
                batch_size=bs,
                length_before_new_iter=list_size
            )
            return sampler, bs
    return PKPolicy()

class BaseDatasetBuilder(ABC):
    _ALLOWED = {"train", "val", "test"}
    _ALIASES = {"validate":"val","validation":"val","dev":"val","train":"train","test":"test"}
    _LOADER_ARG_KEYS = {
        "batch_size", "num_workers", "pin_memory", "drop_last",
        "prefetch_factor", "persistent_workers", "sampler", "shuffle",
        "timeout", "generator", "worker_init_fn", "collate_fn"
    }

    def __init__(self, config: DictConfig):
        self.config = config
        self._datasets: Dict[str, Dataset] = {}
        self._loaders: Dict[str, DataLoader] = {}
        self.logger = get_logger()
        
        # ---- common DataLoader configuration (moved to Base) ----
        tcfg = get_config(config, "training", OmegaConf.create({}))
        self.batch_size: int = int(get_config(tcfg, "batch_size", 32))
        self.eval_batch_size: int = int(get_config(tcfg, "eval_batch_size", self.batch_size))
        self.num_workers: int = int(get_config(tcfg, "num_workers", 4))
        self.pin_memory: bool = bool(get_config(tcfg, "pin_memory", True))
        self.persistent_workers: bool = bool(get_config(tcfg, "persistent_workers", self.num_workers > 0))
        self.prefetch_factor: Optional[int] = get_config(tcfg, "prefetch_factor", None)
        self.timeout: float = float(get_config(tcfg, "timeout", 0))
        self._deterministic: bool = bool(get_config(tcfg, "deterministic", False))
        self._seed: Optional[int] = get_config(tcfg, "seed", None)

        self._sampler_policy = self._init_sampler_policy()

        self._generator = None
        self._worker_init_fn = None
        if self._deterministic and self._seed is not None:
            g = torch.Generator()
            g.manual_seed(int(self._seed))
            self._generator = g
            def _init_fn(worker_id):
                s = int(self._seed) + worker_id
                random.seed(s); np.random.seed(s); torch.manual_seed(s)
            self._worker_init_fn = _init_fn

    def _init_sampler_policy(self) -> SamplerPolicy:
        scfg = get_config(self.config, "training.sampler", OmegaConf.create({}))
        name = str(get_config(scfg, "name", "auto")).lower()
        if name == "auto":
            name = self.default_sampler_name()
        if name == "pk":
            p = int(get_config(scfg, "p", 16)); k = int(get_config(scfg, "k", 4))
            return _make_pk_policy(p, k)
        if name == "random":
            return RandomPolicy(self.batch_size)
        if name == "sequential":
            class SeqPolicy(SamplerPolicy):
                def build(_, dataset, split): return SequentialSampler(dataset), (self.batch_size if split=="train" else None)
            return SeqPolicy()
        raise ValueError(f"Unknown sampler policy: {name}")

    def default_sampler_name(self) -> str:
        return "random"

    def _normalize_split(self, split: str) -> str:
        s = self._ALIASES.get((split or "").strip().lower(), split)
        if s not in self._ALLOWED: raise ValueError(f"Unsupported split '{split}'. Allowed: {sorted(self._ALLOWED)}")
        return s

    def get_dataset(self, split: str, **overrides) -> Dataset:
        if overrides: return self.build_dataset(split, **overrides)
        if split not in self._datasets: self._datasets[split] = self.build_dataset(split)
        return self._datasets[split]

    def get_loader(self, split: str, **overrides) -> DataLoader:
        split = self._normalize_split(split)
        if overrides:
            # --- split overrides ---
            dataset_overrides = {k: v for k, v in overrides.items()
                                 if k not in self._LOADER_ARG_KEYS}
            loader_overrides  = {k: v for k, v in overrides.items()
                                 if k in self._LOADER_ARG_KEYS and v is not None}

            ds: Dataset = overrides.get("dataset") or self.build_dataset(split, **dataset_overrides)

            args = self.default_loader_args(split, ds)
            collate_fn = loader_overrides.pop("collate_fn", None)
            args.update(loader_overrides)
            if collate_fn is not None:
                args["collate_fn"] = collate_fn

            return DataLoader(ds, **args)

        if split not in self._loaders:
            ds = self.get_dataset(split)
            args = self.default_loader_args(split, ds)
            self._loaders[split] = DataLoader(ds, **args)
        return self._loaders[split]

    def default_loader_args(self, split: str, dataset: Dataset) -> Dict[str, Any]:
        split = self._normalize_split(split)
        sampler, bs_override = self._sampler_policy.build(dataset, split)
        is_train = (split == "train")
        batch_size = bs_override or (self.batch_size if is_train else self.eval_batch_size)
        args: Dict[str, Any] = dict(
            batch_size=batch_size,
            shuffle=is_train and (sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=is_train,
            persistent_workers=(self.num_workers > 0 and self.persistent_workers),
            timeout=self.timeout,
            generator=self._generator,
            worker_init_fn=self._worker_init_fn,
        )
        if self.prefetch_factor is not None and self.num_workers > 0:
            args["prefetch_factor"] = int(self.prefetch_factor)
        return args

    def resolve_collate_fn(self, split: str, dataset: Dataset):
        return None

    @abstractmethod
    def build_dataset(self, split: str, **overrides) -> Dataset: ...

class BaseUEBuilder(BaseDatasetBuilder):
    """
    General UE builder:
      - train: ConcatDataset(UEKey(train_clean), UEKey(val_clean))
      - val  : None (no online validation during UE training phase)
      - test : Reuse the clean test from the base task builder
    By default, the transforms for the UE phase are overridden as:
      - train: normalize=False, geom_aug=False (keep Resize/ToTensor, do not Normalize or apply geometric augmentation)
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._base_builder_name = str(get_config(config, "ue.base_task_builder", "mimic_cxr"))

    def _base(self):
        return get_dataset_builder(self._base_builder_name)(self.config)


    def _key_spec(self) -> DictConfig:
        return require_config(self.config, "ue.key", type_=DictConfig)

    def _ue_default_overrides(self, split: str) -> Dict[str, Any]:
        """
        Default transform overrides for the UE phase; if the caller explicitly provides transform/normalize/geom_aug, those take precedence.
        - Training: do not Normalize (because we add noise to x and then Normalize inside the wrapper/trainer), do not apply geometric augmentation
        - Others: keep the base builder defaults (do not force here)
        """
        split = split.lower()
        if split == "train":
            return {"normalize": False, "geom_aug": False}
        return {}

    def _merge_overrides(self, user: Dict[str, Any], ue_defaults: Dict[str, Any]) -> Dict[str, Any]:
        # Explicitly provided transform/normalize/geom_aug by the user take precedence
        out = dict(ue_defaults)
        for k, v in (user or {}).items():
            out[k] = v
        return out

    def build_dataset(self, split: str, **overrides) -> Optional[Dataset]:
        split = self._normalize_split(split)
        base = self._base()

        # Merge UE default overrides
        merged = self._merge_overrides(overrides, self._ue_default_overrides(split))

        if split == "train":
            ds_tr = base.get_dataset("train", **merged)  # clean
            ds_va = base.get_dataset("val",   **merged)  # clean

            # Delayed import to avoid circular dependency
            from .uekey_dataset import UEKeyDataset

            key_spec = self._key_spec()
            ds_tr_k = UEKeyDataset(ds_tr, key_spec)
            ds_va_k = UEKeyDataset(ds_va, key_spec)
            return UEConcatDataset([ds_tr_k, ds_va_k])

        if split == "val":
            return None

        if split == "test":
            return base.get_dataset("test", **merged)

        raise ValueError(f"Unsupported split '{split}' for UE builder.")

    def get_loader(self, split: str, **overrides):
        split = self._normalize_split(split)
        if split == "val":
            return None
        # Also inject UE default overrides into the DataLoader construction path (so the caller can override)
        merged = self._merge_overrides(overrides, self._ue_default_overrides(split))
        return super().get_loader(split, **merged)


def _attach_dataset_task(config: DictConfig, value: str) -> DictConfig:
    # Make a non-struct copy (preserve interpolations)
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=False))

    # Ensure dataset node exists and is writable
    if OmegaConf.select(cfg, "dataset") is None:
        cfg.dataset = OmegaConf.create({})
    OmegaConf.set_struct(cfg.dataset, False)

    cfg.dataset.task = value  # safe to set
    return cfg