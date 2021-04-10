"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from typing import Dict, Any, List
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import logging
import time

from torch.optim.lr_scheduler import LambdaLR, StepLR

import torch.multiprocessing as mp
import torch.distributed as dist
import pandas as pd
import numpy as np
import torch
import tqdm

from ..data.utils import extend_vocabulary, load_data, right_shift
from ..models import get_tokenizer, get_model
from ..utils.logging import logger, mp_logger
from ..optim import get_optimizer_scheduler
from ..utils.io import create_dir


class TrainingAgent:
    def __init__(self, args: Namespace, log_q=None) -> None:
        if log_q is not None:
            logger.addHandler(logging.handlers.QueueHandler(log_q))
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
        self.epoch = 0
        self.trial = 0

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
        # if self.args.rank not in set([-1, 0]):
        #     dist.barrier()
        self.tokenizer = get_tokenizer(self.args.train.tokenizer,
                                       self.args.io.cache_dir,
                                       dataset=self.args.data.name,
                                       task=self.args.data.task)
        logger.info("Extending vocab...")
        extend_vocabulary(self.tokenizer, fname=Path(self.args.data.vocab))
        # extend_vocabulary(self.tokenizer, fname=Path(valid_src))
        logger.info("Loading train dataset...")
        self.train_loader, self.train_sampler = load_data(
            fname=Path(self.args.data.train_src),
            tokenizer=self.tokenizer,
            cache_dir=self.args.io.cache_dir,
            task=self.args.data.task,
            max_samples=self.args.data.max_train_samples,
            max_length=self.args.train.max_length,
            batch_size=self.args.train.batch_size,
            num_workers=self.args.data.num_workers,
            overwrite_cache=self.args.data.overwrite_cache,
            split='train',
            distributed=self.args.distributed)
        logger.info("Loading validation dataset...")
        self.val_loader, self.val_sampler = load_data(
            fname=Path(self.args.data.val_src),
            tokenizer=self.tokenizer,
            cache_dir=self.args.io.cache_dir,
            task=self.args.data.task,
            max_samples=self.args.data.max_val_samples,
            max_length=self.args.train.max_length,
            batch_size=self.args.train.batch_size,
            num_workers=self.args.data.num_workers,
            overwrite_cache=self.args.data.overwrite_cache,
            split='val',
            distributed=self.args.distributed)
        # if self.args.rank == 0:
        #     dist.barrier()

    def reset(self) -> None:
        logger.info("Starting trial reset...")
        logger.info("Loading model...")
        # if self.args.rank not in set([-1, 0]):
        #     dist.barrier()
        if self.args.io.resume is not None:
            self.load_checkpoint(Path(self.args.io.resume))
        else:
            self.model = get_model(self.args.train.model,
                                   cache_dir=self.args.io.cache_dir,
                                   tokenizer_len=len(self.tokenizer),
                                   pretrained=self.args.train.pretrained,
                                   task=self.args.data.task)
            self.optimizer, self.scheduler = get_optimizer_scheduler(
                optim_method=self.args.train.optimizer,
                scheduler_method=self.args.train.scheduler,
                net_parameters=self.model.parameters(),
                max_epochs=self.args.train.max_epochs,
                train_loader_len=len(self.train_loader),
                optimizer_kwargs=self.args.train.optimizer_kwargs,
                scheduler_kwargs=self.args.train.scheduler_kwargs)
        # self.model.resize_token_embeddings(len(self.tokenizer))
        # if self.args.rank == 0:
        #     dist.barrier()
        # self.model.to(self.device)
        if self.args.distributed:
            if self.gpu is not None:
                self.model.to(self.device)
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[self.gpu],
                    find_unused_parameters=True)
            else:
                self.model.cuda()
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    find_unused_parameters=True)
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
        # self.loss = torch.nn.CrossEntropyLos() if self.args.train.loss ==\
        #     'cross_entropy' else None
        # self.loss.to(self.device)
        logger.info("Done reset")

    def save_stats(self) -> None:
        pd.DataFrame(data=self.stats).to_csv(self.output_dir / 'stats.csv')

    @staticmethod
    def unwrap(model: torch.nn.Module) -> torch.nn.Module:
        if hasattr(model, "module"):
            return TrainingAgent.unwrap(model.module)
        else:
            return model

    def load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Unknown resume. Starting from scratch.")
            self.args.io.resume = None
            self.reset()
            return
        self.args.train.pretrained = True
        train_state = torch.load(str(path / 'train_state.pt'))
        # train_state['args'].io.resume = self.args.io.resume
        # if self.args != train_state['args']:
        #     raise ValueError("Checkpoint args and config args don't match")
        self.model = get_model(
            self.args.train.model,
            cache_dir=self.args.io.cache_dir,
            tokenizer_len=len(self.tokenizer),
            pretrained=True,
            weights=str(path),
            task=self.args.data.task)
        self.optimizer, self.scheduler = get_optimizer_scheduler(
            optim_method=self.args.train.optimizer,
            scheduler_method=self.args.train.scheduler,
            net_parameters=self.model.parameters(),
            max_epochs=self.args.train.max_epochs,
            train_loader_len=len(self.train_loader),
            optimizer_kwargs=self.args.train.optimizer_kwargs,
            scheduler_kwargs=self.args.train.scheduler_kwargs)
        self.optimizer.load_state_dict(train_state['optimizer'])
        self.scheduler.load_state_dict(train_state['scheduler'])
        self.epoch = train_state['epoch']
        self.trial = train_state['trial']
        self.args.io.resume = None

    def save_checkpoint(self, trial: int, epoch: int, stamp: str) -> None:
        model_to_save = TrainingAgent.unwrap(self.model)
        if hasattr(model_to_save, 'save_pretrained'):
            model_to_save.save_pretrained(
                str(self.checkpoint_dir / stamp))
        else:
            torch.save(model_to_save.state_dict(),
                       str(self.checkpoint_dir / stamp))
        torch.save({
            'args': self.args,
            'epoch': epoch,
            'trial': trial,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(), },
            str(self.checkpoint_dir / stamp / 'train_state.pt'))
        # torch.save(self.optimizer.state_dict(), str(
        #     self.checkpoint_dir / folder / 'optimizer.pt'))
        # if self.scheduler is not None:
        #     torch.save(self.optimizer.state_dict(), str(
        #         self.checkpoint_dir / folder / 'scheduler.pt'))

    def train(self) -> None:
        for trial in range(self.args.train.num_trials):
            self.reset()
            if trial == self.args.train.num_trials - self.trial:
                break
            self.run_epochs(trial + self.trial, range(
                self.epoch, self.args.train.max_epochs))
            self.epoch = 0
            if not self.args.mpd or (
                self.args.mpd and
                    self.args.rank % self.args.ngpus_per_node == 0):
                self.save_checkpoint(trial=trial,
                                     epoch=self.args.train.max_epochs - 1,
                                     stamp=f'final-{trial}')

    def run_epochs(self, trial: int, epochs: List[int]) -> None:
        # total_steps = len(self.train_loader) * self.config['max_epochs']
        best_loss = 0
        for epoch in epochs:
            if self.args.distributed:
                self.train_sampler.set_epoch(epoch)
            start_time = time.time()
            train_loss = self.epoch_iteration(trial, epoch)
            epoch_time = time.time()
            val_loss = self.validate(trial, epoch)
            if isinstance(self.scheduler, StepLR):
                self.scheduler.step()
            end_time = time.time()
            logger.info(f"T {trial} | E {epoch} | " +
                        f"E time: {epoch_time - start_time} | " +
                        f"T time {end_time - epoch_time} | " +
                        f"Train Loss {train_loss} | " +
                        f"Val Loss {val_loss}")
            self.stats[epoch] = {
                'train_loss': train_loss, 'val_loss': val_loss}
            if (epoch % self.args.io.save_freq == 0 and epoch > 0):
                if not self.args.mpd or (
                    self.args.mpd and
                        self.args.rank % self.args.ngpus_per_node == 0):
                    self.save_checkpoint(trial=trial,
                                         epoch=epoch,
                                         # stamp=f"-t{trial}-e{epoch}")
                                         stamp="latest")
            if np.greater(val_loss, best_loss):
                best_loss = val_loss
                if not self.args.mpd or (
                    self.args.mpd and
                        self.args.rank % self.args.ngpus_per_node == 0):
                    self.save_checkpoint(trial=trial,
                                         epoch=epoch,
                                         stamp="best")
            self.save_stats()

    def epoch_iteration(self, trial: int, epoch: int) -> float:
        self.model.train()
        train_loss = 0
        for step, batch in enumerate(
                tqdm.tqdm(self.train_loader,
                          desc=f'Trial {trial} | Epoch {epoch} | Train')):
            inputs = batch['input_ids'].to(
                self.device, non_blocking=True)
            labels = inputs
            attention_mask = None
            if 'labels' in batch.keys():
                labels = batch['labels'].to(
                    self.device, non_blocking=True)
            if 'attention_mask' in batch.keys():
                attention_mask = batch['attention_mask'].to(
                    self.device, non_blocking=True)
            self.optimizer.zero_grad()
            if self.args.train.model == 'bert-base-uncased' and \
                    self.args.data.task == 'nmt':
                loss = self.model(
                    input_ids=inputs,
                    decoder_input_ids=right_shift(
                        self.tokenizer.pad_token_id,
                        self.tokenizer.pad_token_id,
                        inputs).to(self.device, non_blocking=True),
                    attention_mask=attention_mask,
                    labels=labels).loss
            else:
                loss = self.model(
                    input_ids=inputs,
                    # decoder_input_ids=right_shift(
                    #     self.tokenizer.pad_token_id,
                    #     self.tokenizer.pad_token_id,
                    #     inputs).to(self.device, non_blocking=True),
                    attention_mask=attention_mask,
                    labels=labels).loss
            del inputs, attention_mask, labels
            # loss = self.criterion(outputs.logits, targets) with labels=None
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
            if isinstance(self.scheduler, LambdaLR):
                self.scheduler.step()
        return train_loss / (step + 1)

    def validate(self, trial: int, epoch: int) -> float:
        self.model.eval()
        val_loss = 0
        for step, batch in enumerate(
                tqdm.tqdm(self.val_loader,
                          desc=f'Trial {trial} | Epoch {epoch} | Validate')):
            inputs = batch['input_ids'].to(
                self.device, non_blocking=True)
            labels = inputs
            attention_mask = None
            if 'labels' in batch.keys():
                labels = batch['labels'].to(
                    self.device, non_blocking=True)
            if 'attention_mask' in batch.keys():
                attention_mask = batch['attention_mask'].to(
                    self.device, non_blocking=True)
            with torch.no_grad():
                loss = self.model(
                    inputs, attention_mask=attention_mask, labels=labels).loss
            del inputs, attention_mask, labels
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
        log_q = None  # mp_logger(args.log)
        args.world_size *= ngpus_per_node
        mp.spawn(worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, deepcopy(args), log_q))
    else:
        worker(args.gpu if not args.cpu else None, ngpus_per_node, args, None)


def worker(gpu: int, ngpus_per_node: int, args: Namespace, log_q):
    args.gpu = gpu
    if args.distributed:
        if args.mpd:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
    training_agent = TrainingAgent(args=args, log_q=log_q)
    training_agent.train()
