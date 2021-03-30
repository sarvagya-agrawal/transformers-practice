"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from argparse import Namespace
from typing import Dict, Any

from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD, AdamW, Adam

from .lambdalr import ConstantLambdaLR, LinearLambdaLR

OPTIMIZERS = ['SGD', 'AdamW', 'Adam']
SCHEDULERS = ['LambdaLR', 'StepLR', 'None', None]


def get_optimizer_scheduler(
        optim_method: str,
        scheduler_method: str,
        net_parameters,
        max_epochs: int,
        train_loader_len: int,
        optimizer_kwargs: Dict[str, Any] = None,
        scheduler_kwargs: Dict[str, Any] = None):
    if optimizer_kwargs is None:
        optimizer_kwargs = dict()
    if scheduler_kwargs is None:
        optimizer_kwargs = dict()
    optimizer = eval(optim_method)(net_parameters, **optimizer_kwargs)
    if scheduler_method == 'LinearLambdaLR':
        scheduler_kwargs['num_training_steps'] = train_loader_len * \
            max_epochs
    scheduler = None if scheduler_method == 'None' else eval(
        scheduler_method)(optimizer, **scheduler_kwargs)
    return optimizer, scheduler
