"""Dataset package initialization and registration."""

from ..registry import register_dataset
from .process3d import BraTSGLI3DDataset, build_brats_gli_3d_loader

from .mimic_cxr import (
    MIMICCXRDataset,
    MIMICCXRReIDDataset,
    MIMICCXRClassificationDataset,
    MIMICCXRUEDataset,
)

from .grape import (
    GrapeDataset,
    GrapeReIDDataset,
    GrapeSegDataset,
    GrapeVFRegDataset,
    GrapeUEDataset,
)

# Import builders so they register themselves
from . import mimic_builders  # noqa: F401
from . import grape_builders  # noqa: F401

# Register dataset implementations with the unified registry
register_dataset('mimic_cxr')(MIMICCXRDataset)
register_dataset('mimic_cxr_reid')(MIMICCXRReIDDataset)
register_dataset('mimic_cxr_cls')(MIMICCXRClassificationDataset)
register_dataset('mimic_cxr_ue')(MIMICCXRUEDataset)
register_dataset('grape')(GrapeDataset)
register_dataset('grape_reid')(GrapeReIDDataset)
register_dataset('grape_seg')(GrapeSegDataset)
register_dataset('grape_vf_reg')(GrapeVFRegDataset)
register_dataset('grape_ue')(GrapeUEDataset)

__all__ = [
    'MIMICCXRDataset',
    'MIMICCXRReIDDataset',
    'MIMICCXRClassificationDataset',
    'MIMICCXRUEDataset',
    'GrapeDataset',
    'GrapeReIDDataset',
    'GrapeSegDataset',
    'GrapeVFRegDataset',
    'GrapeUEDataset',
]
