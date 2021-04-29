"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from typing import Dict, Any

import math

from transformers import AdamW, get_scheduler
from adas import Adas


def get_optimizer_scheduler(
        optim_method: str,
        scheduler_method: str,
        params,
        named_parameters,
        max_epochs: int,
        max_train_steps: int,
        optimizer_kwargs: Dict[str, Any] = None,
        scheduler_kwargs: Dict[str, Any] = None):
    if optimizer_kwargs is None:
        optimizer_kwargs = dict()
    if scheduler_kwargs is None:
        optimizer_kwargs = dict()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in named_parameters if not any(
                nd in n for nd in no_decay)],
            "weight_decay": optimizer_kwargs['weight_decay'] if
            'weight_decay' in optimizer_kwargs.keys() else 0.0
        },
        {
            "params": [p for n, p in named_parameters if any(
                nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if optim_method == 'Adas':
        optimizer_grouped_parameters.append({
            'all_params': params})
        optimizer_kwargs['params_dict'] = True
    optimizer = eval(optim_method)(optimizer_grouped_parameters,
                                   **optimizer_kwargs)

    scheduler = None
    if scheduler_method is not None:
        scheduler = get_scheduler(
            name=scheduler_method,
            optimizer=optimizer,
            num_training_steps=max_train_steps,
            **scheduler_kwargs)
    return optimizer, scheduler
