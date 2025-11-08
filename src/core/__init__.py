"""
核心模块初始化
"""

from .trainer_base import TrainerBase, HookBase
from .trainers import CLSTrainer, REIDTrainer
from .trainer import Trainer
from .hooks import TimerHook, CheckpointHook, LearningRateSchedulerHook
from .experiment_manager import ExperimentManager

__all__ = [
    'Trainer', 
    'TrainerBase',
    'HookBase',
    'CLSTrainer',
    'REIDTrainer',
    'TimerHook',
    'CheckpointHook',
    'LearningRateSchedulerHook', 
    'ExperimentManager',
]
