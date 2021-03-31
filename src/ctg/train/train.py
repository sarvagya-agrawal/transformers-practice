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


import torch.multiprocessing as mp
import torch.distributed as dist
import pandas as pd
import torch
import tqdm

from ..models import get_tokenizer, get_model
from ..optim import get_optimizer_scheduler
from ..utils.logging import logger
from ..data.utils import load_data
from ..utils.io import create_dir


class TrainingAgent:
    def __init__(self, args: Namespace,) -> None:
        self.train_loader = None
        self.train_sampler = None
        self.val_loader = None
        self.val_sampler = None
        self.model: torch.nn.Module = None
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler = None
        self.loss = None
        self.output_filename: Path = None
        self.checkpoint = None
        self.stats: Dict[str, Any] = dict()

        self.args = args
        if self.args.mpd:
            self.gpu = self.args.rank
            self.device = torch.device('cuda', self.gpu)
        elif isinstance(args.gpu, list):
            self.gpu = args.gpu
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.gpu = args.gpu
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_dir = create_dir(Path(self.args.io.output))
        self.checkpoint_dir = create_dir(Path(self.args.io.checkpoint))
        self.setup()
        logger.info("Training Agent initialized")

    def setup(self) -> None:
        if self.args.distributed:
            self.args.train.batch_size = int(self.args.train.batch_size /
                                             self.args.ngpus_per_node)
            self.args.data.num_workers = int((self.args.data.num_workers +
                                              self.args.ngpus_per_node - 1) /
                                             self.args.ngpus_per_node)

        logger.info(f"Grabbing tokenizer: {self.args.train.tokenizer}")
        self.tokenizer = get_tokenizer(self.args.train.tokenizer,
                                       self.args.io.cache_dir)
        self.task_setup()
        logger.info("Loading train dataset...")
        self.train_loader, self.train_sampler = load_data(
            src_fname=Path(self.args.data.train_src),
            tgt_fname=Path(self.args.data.train_tgt) if self.args.data.val_tgt
            is not None else None,
            tokenizer=self.tokenizer,
            cache_dir=self.args.io.cache_dir,
            max_length=self.args.train.max_length,
            batch_size=self.args.train.batch_size,
            num_workers=self.args.data.num_workers,
            overwrite_cache=self.args.data.overwrite_cache,
            split='train',
            distributed=self.args.distributed)
        logger.info("Loading validation dataset...")
        self.val_loader, self.val_sampler = load_data(
            src_fname=Path(self.args.data.val_src),
            tgt_fname=Path(self.args.data.val_tgt) if self.args.data.val_tgt
            is not None else None,
            tokenizer=self.tokenizer,
            cache_dir=self.args.io.cache_dir,
            max_length=self.args.train.max_length,
            batch_size=self.args.train.batch_size,
            num_workers=self.args.data.num_workers,
            overwrite_cache=self.args.data.overwrite_cache,
            split='val',
            distributed=self.args.distributed)

    def reset(self) -> None:
        logger.info("Starting trial reset...")
        logger.info("Loading model...")
        self.model = get_model(self.args.train.model,
                               cache_dir=self.args.io.cache_dir,
                               pretrained=self.args.train.pretrained)
        # self.model.to(self.device)
        if self.args.distributed:
            if self.gpu is not None:
                self.model.to(self.device)
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[self.gpu])
            else:
                self.model.cuda()
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model)
        elif isinstance(self.gpu, list):
            self.model = torch.nn.DataParallel(self.model)
            self.model.to(self.device)
        elif isinstance(self.gpu, int):
            torch.cuda.set_device(self.gpu)
            self.model = self.model.cuda(self.gpu)
        logger.info(f"Setting model to {self.device}")
        # if len(self.args.train.gpu) > 1:
        #     self.model = torch.nn.DataParallel(self.model)
        logger.info(
            "Grabbing optimizer and scheduler: " +
            f"{self.args.train.optimizer} - {self.args.train.scheduler}")
        self.optimizer, self.scheduler = get_optimizer_scheduler(
            optim_method=self.args.train.optimizer,
            scheduler_method=self.args.train.scheduler,
            net_parameters=self.model.parameters(),
            max_epochs=self.args.train.max_epochs,
            train_loader_len=len(self.train_loader),
            optimizer_kwargs=self.args.train.optimizer_kwargs,
            scheduler_kwargs=self.args.train.scheduler_kwargs)

        # self.loss = torch.nn.CrossEntropyLos() if self.args.train.loss ==\
        #     'cross_entropy' else None
        # self.loss.to(self.device)
        logger.info("Done reset")

    def task_setup(self) -> None:
        if self.args.data.name == 'multiwoz2.1':
            if self.args.data.task == 'causal':
                self.tokenizer.add_special_tokens(
                    {'bos_token': '<|endoftext|>'})
                self.tokenizer.add_special_tokens(
                    {'eos_token': '<|endoftext|>'})
            if self.tokenizer._pad_token is None:
                # tokenizer.add_special_tokens(
                #     {'pad_token': '[PAD]'})
                self.tokenizer.pad_token = 0.

    def save_stats(self) -> None:
        pd.DataFrame(data=self.stats).to_csv(self.output_dir / 'stats.csv')

    @staticmethod
    def unwrap(model: torch.nn.Module) -> torch.nn.Module:
        if hasattr(model, "module"):
            TrainingAgent.unwrap(model)
        else:
            return model

    def save_checkpoint(self, stamp: str = '') -> None:
        model_to_save = TrainingAgent.unwrap(self.model)
        if hasattr(model_to_save, 'save_pretrained'):
            model_to_save.save_pretrained(
                str(self.checkpoint_dir / f'model-check{stamp}'))
        else:
            torch.save(model_to_save.state_dict(),
                       str(self.checkpoint_dir / f'model-check{stamp}'))
        torch.save({'args': self.args},
                   str(self.checkpoint_dir / f'args{stamp}.json'))
        torch.save(self.optimizer.state_dict(), str(
            self.checkpoint_dir / f'optimizer{stamp}.pt'))
        if self.scheduler is not None:
            torch.save(self.optimizer.state_dict(), str(
                self.checkpoint_dir / f'scheduler{stamp}.pt'))

    def train(self) -> None:
        for trial in range(self.args.train.num_trials):
            self.reset()
            self.run_epochs(trial, range(0, self.args.train.max_epochs))
        if not self.args.mpd or (
            self.args.mpd and
                self.args.rank % self.args.ngpus_per_node == 0):
            self.save_checkpoint(stamp='-final')

    def run_epochs(self, trial: int, epochs: List[int]) -> None:
        # total_steps = len(self.train_loader) * self.config['max_epochs']
        for epoch in epochs:
            if self.args.distributed:
                self.train_sampler.set_epoch(epoch)
            start_time = time.time()
            train_loss = self.epoch_iteration(trial, epoch)
            epoch_time = time.time()
            val_loss = self.validate(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            end_time = time.time()
            logger.info(f"E time: {epoch_time - start_time} | "
                        f"T time {end_time - epoch_time} | " +
                        f"Train Loss {train_loss} | " +
                        f"Val Loss {val_loss}")
            self.stats[epoch] = {
                'train_loss': train_loss, 'val_loss': val_loss}
            if epoch % self.args.io.save_freq == 0 and epoch > 0:
                if not self.args.mpd or (
                    self.args.mpd and
                        self.args.rank % self.args.ngpus_per_node == 0):
                    self.save_checkpoint(f"-{epoch}")
            self.save_stats()

    def epoch_iteration(self, trial: int, epoch: int) -> float:
        self.model.train()
        train_loss = 0
        for step, batch in enumerate(
                tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch} | Train')):
            if self.gpu is not None:
                if 'attention_mask' not in batch.keys():
                    inputs = batch['input_ids'].to(
                        self.device, non_blocking=True)
                    attention_mask = None
                    targets = inputs
                elif 'attention_mask' in batch.keys():
                    inputs = batch['input_ids'].to(
                        self.device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(
                        self.device, non_blocking=True)
                    targets = inputs
                elif 'labels' in batch.keys():
                    inputs = batch['input_ids'].to(
                        self.device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(
                        self.device, non_blocking=True)
                    targets = batch['labels'].to(
                        self.device, non_blocking=True)
            self.optimizer.zero_grad()
            outputs = self.model(
                inputs,
                attention_mask=attention_mask,
                labels=targets)
            del inputs, attention_mask, targets
            loss = outputs[0]
            # loss = self.criterion(outputs, targets)
            if isinstance(self.gpu, list):
                loss = loss.mean()
            # if self.args.train.gradient_accumulation_steps > 1:
            # loss /= self.args.train.gradient_accumulation_steps
            train_loss += loss.item()
            loss.backward()
            if self.args.train.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.args.train.clip_grad)
            self.optimizer.step()
        return train_loss / (step + 1)

    def validate(self, epoch: int) -> float:
        self.model.eval()
        val_loss = 0
        for step, batch in enumerate(
                tqdm.tqdm(self.val_loader, desc=f'Epoch {epoch} | Validate')):
            if self.gpu is not None:
                if 'attention_mask' not in batch.keys():
                    inputs = batch['input_ids'].to(
                        self.device, non_blocking=True)
                    attention_mask = None
                    targets = inputs
                elif 'attention_mask' in batch.keys():
                    inputs = batch['input_ids'].to(
                        self.device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(
                        self.device, non_blocking=True)
                    targets = inputs
                elif 'labels' in batch.keys():
                    inputs = batch['input_ids'].to(
                        self.device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(
                        self.device, non_blocking=True)
                    targets = batch['labels'].to(
                        self.device, non_blocking=True)
            with torch.no_grad():
                outputs = self.model(
                    inputs, attention_mask=attention_mask, labels=targets)
            del inputs, attention_mask, targets
            loss = outputs[0]
            if isinstance(self.gpu, list):
                loss = loss.mean()
            val_loss += loss.item()
        return val_loss / (step + 1)


def main(args: Namespace) -> None:
    args.ngpus_per_node = ngpus_per_node = torch.cuda.device_count() if \
        args.gpu is None else 1 if isinstance(args.gpu, int) else len(args.gpu)
    args.distributed = (args.mpd or args.world_size > 1) and not args.cpu
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"NGPUS: {ngpus_per_node}")
    if args.mpd and not args.cpu:
        args.world_size *= ngpus_per_node
        mp.spawn(worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, deepcopy(args)))
    else:
        worker(args.gpu if not args.cpu else None, ngpus_per_node, args)


def worker(gpu: int, ngpus_per_node: int, args: Namespace):
    args.gpu = gpu
    if args.distributed:
        if args.mpd:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
    training_agent = TrainingAgent(args=args)
    training_agent.train()
