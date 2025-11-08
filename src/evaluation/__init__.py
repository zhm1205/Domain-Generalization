"""Evaluation strategy package."""

# Import strategies so that they register themselves with the global registry
from . import mimic_cxr_eval  # noqa: F401
from . import reid_eval  # noqa: F401
from . import grape_eval  # noqa: F401
from . import seg_eval_dice


__all__ = [
    'mimic_cxr_eval',
    'reid_eval',
    'grape_eval',
]
