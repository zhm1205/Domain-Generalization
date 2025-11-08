"""
训练器模块入口
"""

from .trainer_base import TrainerBase, HookBase
from .trainers import CLSTrainer
from .hooks import TimerHook, CheckpointHook, LearningRateSchedulerHook, MetricsLoggerHook

# 保持向后兼容
Trainer = CLSTrainer

__all__ = [
    'Trainer', 
    'TrainerBase',
    'CLSTrainer', 
    'HookBase',
    'TimerHook',
    'CheckpointHook', 
    'LearningRateSchedulerHook',
    'MetricsLoggerHook'
]
