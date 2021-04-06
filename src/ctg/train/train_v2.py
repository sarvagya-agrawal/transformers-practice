"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from typing import Dict, Any, List
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import time

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

import torch.multiprocessing as mp
import torch.distributed as dist
import pandas as pd
import numpy as np
import torch
import tqdm

from ..models import get_tokenizer, get_model
from ..optim import get_optimizer_scheduler
from ..utils.logging import logger
from ..data.utils import load_data
from ..utils.io import create_dir


def main(args: Namespace) -> None:
    tokenizer = get_tokenizer(args.train.tokenizer,
                              args.io.cache_dir,
                              dataset=args.data.name,
                              task=args.data.task)
    train_set = load_data(
        src_fname=Path(args.data.train_src),
        tokenizer=tokenizer,
        cache_dir=args.io.cache_dir,
        max_length=args.train.max_length,
        batch_size=args.train.batch_size,
        num_workers=args.data.num_workers,
        overwrite_cache=args.data.overwrite_cache,
        split='train',
        distributed=False)
    val_set = load_data(
        src_fname=Path(args.data.val_src),
        tokenizer=tokenizer,
        cache_dir=args.io.cache_dir,
        max_length=args.train.max_length,
        batch_size=args.train.batch_size,
        num_workers=args.data.num_workers,
        overwrite_cache=args.data.overwrite_cache,
        split='val',
        distributed=False)
    model = get_model(args.train.model,
                      cache_dir=args.io.cache_dir,
                      pretrained=args.train.pretrained,
                      task=args.data.task)
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.io.output,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.train.batch_size,
        per_device_eval_batch_size=args.train.batch_size,
        num_train_epochs=args.train.max_epochs,)

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,)
    results = trainer.train()
    trainer.save_model()
    metrics = results.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state("train")

    metrics = trainer.evaluate(
        max_length=args.max_length, num_beams=5, metric_key_prefix='eval')
    trainer.log_metrics('eval', metrics)
    trainer.save_metrics('eval', metrics)
