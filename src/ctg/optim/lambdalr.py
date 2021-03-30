"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from typing import Dict, Any
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def ConstantLambdaLR(optimizer: Optimizer,
                     num_warmup_steps: int,
                     last_epoch: int = -1,
                     kwargs: Dict[str, Any] = None) -> LambdaLR:
    if kwargs is None:
        kwargs = dict()

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch, **kwargs)


def LinearLambdaLR(optimizer: Optimizer,
                   num_warmup_steps: int,
                   num_training_steps: int,
                   last_epoch: int = -1,
                   kwargs: Dict[str, Any] = None) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
