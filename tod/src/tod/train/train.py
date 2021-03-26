"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from typing import Union, Dict, Any, List
from pathlib import Path

import time

from transformers import PreTrainedModel as HFPretrained
from configargparse import Namespace

import torch

from .models import get_tokenizer, get_model
from .optim import get_optimizer_scheduler
from .data.utils import load_data
from .io import create_dir


class TrainingAgent:
    config: Dict[str, Any] = None
    train_loader = None
    val_loader = None
    model: torch.nn.Module = None
    optimizer: torch.optim.Optimizer = None
    scheduler = None
    loss = None
    output_filename: Path = None
    checkpoint = None

    def __init__(self,
                 model: Union[torch.nn.Module, HFPretrained],
                 args: Namespace,) -> None:
        self.model = model
        self.args = args
        self.reset()

    def reset(self) -> None:
        self.train_loader = load_data(src_fname=self.args.data.train_src,
                                      tgt_fname=self.args.data.train_tgt,
                                      tokenizer=get_tokenizer(self.tokenizer),
                                      max_length=self.args.train.max_length,
                                      batch_size=self.args.train.batch_size,
                                      num_workers=self.args.train.num_workers)
        self.val_loader = load_data(src_fname=self.args.data.val_src,
                                    tgt_fname=self.args.data.val_tgt,
                                    tokenizer=get_tokenizer(self.tokenizer),
                                    max_length=self.args.train.max_length,
                                    batch_size=self.args.train.batch_size,
                                    num_workers=self.args.train.num_workers)
        self.model = get_model(self.args.model)
        self.optimizer, self.scheduler = get_optimizer_scheduler(
            optimizer=self.args.optimizer,
            scheduler=self.args.scheduler,
            optimizer_kwargs=self.args.optimizer_kwargs,
            scheduler_kwargs=self.args.scheduler_kwargs,)
        self.loss = self.get_loss(self.args.train.loss)
        self.output_dir = create_dir(self.args.io.output)
        self.checkpoint_dir = create_dir(self.args.io.checkpoint)

    def train(self) -> None:
        for trial in range(self.args['num_trials']):
            self.reset()
            self.run_epochs(trial, range(0, self.config['max_epochs']))

    def run_epochs(self, trial: int, epochs: List[int]) -> None:
        # total_steps = len(self.train_loader) * self.config['max_epochs']
        for epoch in epochs:
            start_time = time.time()
            train_loss, tain_acc = self.epoch_iteration(trial, epoch)
            epoch_time = time.time()
            test_loss, val_acc = self.evaluate(epoch)
            end_time = time.time()

    def epoch_iteration(self, trial: int, epoch: int) -> None:
        self.model.train()
        train_loss = 0
        for step, (inputs, mask, targets) in enumerate(self.train_loader):
            if self.args['gpu'] is not None and self.device == 'cuda':
                inputs = inputs.cuda(self.gpu, non_blocking=True)
                input_mask = mask.cuda(self.gpu, non_blocking=True)
                targets = targets.cuda(self.gpu, non_blocking=True)
            self.optimizer.zero_grad()
            loss, outputs = self.model(
                inputs, attention_mask=input_mask, targets=targets)
            # loss = self.criterion(outputs, targets)
            if self.args['num_gpu'] > 1:
                loss = loss.mean()
            train_loss += loss.item()
            loss.backward()
            if self.args.train.clip_grad > 0:
                torch.nn.utils.clip_grad_norm(self.model.parameters(),
                                              self.args.train.clip_grad)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

    def validate(self, epoch: int):
        self.model.eval()
        test_loss = 0
        for inputs, mask, targets in self.train_loader:
            if self.gpu is not None and self.device == 'cuda':
                inputs = inputs.cuda(self.gpu, non_blocking=True)
                input_mask = mask.cuda(self.gpu, non_blocking=True)
                targets = targets.cuda(self.gpu, non_blocking=True)
            with torch.no_grad():
                loss, outputs = self.model(
                    inputs, attention_mask=input_mask, targets=targets)


def main(args: Namespace) -> None:
    tokenizer = get_tokenizer(args.tokenizer)
    model = get_model(args.model)
    train_loader = load_data(args.train_src, args.train_tgt, tokenizer,
                             max_length=args.max_length,
                             batch_size=args.batch_size)
    val_loader = load_data(args.val_src, args.val_tgt, tokenizer,
                           max_length=args.max_length,
                           batch_size=args.batch_size)
