"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from configargparse import Namespace
from typing import Dict, Any

from torch.optim import SGD, AdamW, Adam


def get_optimizer_scheduler(
        optim_method: str,
        scheduler_method: str,
        net_parameters,
        max_epochs: int,
        train_loader_len: int,
        optimizer_kwargs: Dict[str, Any] = dict(),
        scheduler_kwargs: Dict[str, Any] = dict()):
    optimizer = eval(optim_method)(net_parameters, **optimizer_kwargs)
    if scheduler_method == 'StepLRWarmup':
        scheduler_kwargs['num_train_steps'] = train_loader_len * \
            max_epochs
    scheduler = None if scheduler_method == 'None' else eval(
        scheduler_method)(net_parameters, **scheduler_kwargs)
    return optimizer, scheduler
