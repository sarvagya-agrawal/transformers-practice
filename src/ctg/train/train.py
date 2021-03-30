"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from typing import Dict, Any, List
from argparse import Namespace
from pathlib import Path

import time


import torch
import tqdm


from ..models import get_tokenizer, get_model
from ..optim import get_optimizer_scheduler
from ..utils.logging import logger
from ..data.utils import load_data
from ..utils.io import create_dir


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

    def __init__(self, args: Namespace,) -> None:
        self.args = args
        self.gpu = self.args.train.gpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_dir = create_dir(Path(self.args.io.output))
        self.checkpoint_dir = create_dir(Path(self.args.io.checkpoint))
        self.setup()
        logger.info("Training Agent initialized")

    def setup(self) -> None:
        logger.info(f"Grabbing tokenizer: {self.args.train.tokenizer}")
        self.tokenizer = get_tokenizer(self.args.train.tokenizer,
                                       self.args.io.cache_dir)
        self.task_setup()
        logger.info("Loading train dataset...")
        self.train_loader = load_data(
            src_fname=Path(self.args.data.train_src),
            tgt_fname=Path(self.args.data.train_tgt) if self.args.data.val_tgt
            is not None else None,
            tokenizer=self.tokenizer,
            cache_dir=self.args.io.cache_dir,
            max_length=self.args.train.max_length,
            batch_size=self.args.train.batch_size,
            num_workers=self.args.train.num_workers,
            overwrite_cache=self.args.data.overwrite_cache,
            split='train')
        logger.info("Loading validation dataset...")
        self.val_loader = load_data(
            src_fname=Path(self.args.data.val_src),
            tgt_fname=Path(self.args.data.val_tgt) if self.args.data.val_tgt
            is not None else None,
            tokenizer=self.tokenizer,
            cache_dir=self.args.io.cache_dir,
            max_length=self.args.train.max_length,
            batch_size=self.args.train.batch_size,
            num_workers=self.args.train.num_workers,
            overwrite_cache=self.args.data.overwrite_cache,
            split='val')

    def reset(self) -> None:
        logger.info("Starting trial reset...")
        logger.info("Loading model...")
        self.model = get_model(self.args.train.model,
                               cache_dir=self.args.io.cache_dir,
                               pretrained=self.args.train.pretrained)
        self.model.to(self.device)
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
        if self.args.data.name == 'MultiWOZ':
            if self.args.data.task == 'causal':
                self.tokenizer.add_special_tokens(
                    {'bos_token': '<|endoftext|>'})
                self.tokenizer.add_special_tokens(
                    {'eos_token': '<|endoftext|>'})

    def train(self) -> None:
        for trial in range(self.args.train.num_trials):
            self.reset()
            self.run_epochs(trial, range(0, self.args.train.max_epochs))
        self.save()

    def save(self) -> None:
        self.model.save_pretrained(str(self.output_dir / 'checkpoint'))
        torch.save(self.optimizer.state_dict(), str(
            self.output_dir / 'optimizer.pt'))

    def run_epochs(self, trial: int, epochs: List[int]) -> None:
        # total_steps = len(self.train_loader) * self.config['max_epochs']
        for epoch in epochs:
            start_time = time.time()
            train_loss = self.epoch_iteration(trial, epoch)
            epoch_time = time.time()
            val_loss = self.validate(epoch)
            end_time = time.time()
            logger.info(f"E time: {epoch_time - start_time} | "
                        f"T time {end_time - epoch_time} | " +
                        f"Train Loss {train_loss} | " +
                        f"Val Loss {val_loss}")

    def epoch_iteration(self, trial: int, epoch: int) -> float:
        self.model.train()
        train_loss = 0
        for step, batch in enumerate(
                tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch} | Train')):
            if self.gpu is not None and self.device == 'cuda':
                if len(batch) == 1:
                    # inputs = torch.LongTensor(batch['input_ids']).cuda(
                    #     self.gpu, non_blocking=True)
                    # input_mask = torch.LongTensor(batch['attention_mask']).cuda(
                    #     self.gpu, non_blocking=True)
                    # targets = inputs
                    inputs = batch.cuda(self.gpu, non_blocking=True)
                    targets = inputs
                if len(batch) == 2:
                    # inputs = torch.LongTensor(batch['input_ids']).cuda(
                    #     self.gpu, non_blocking=True)
                    # input_mask = torch.LongTensor(batch['attention_mask']).cuda(
                    #     self.gpu, non_blocking=True)
                    # targets = inputs
                    inputs = batch[0].cuda(self.gpu, non_blocking=True)
                    input_mask = batch[1].cuda(self.gpu, non_blocking=True)
                    targets = inputs
                else:
                    # inputs = torch.LongTensor(batch['input_ids']).cuda(
                    #     self.gpu, non_blocking=True)
                    # input_mask = torch.LongTensor(batch['attention_mask']).cuda(
                    #     self.gpu, non_blocking=True)
                    # targets = torch.LongTensor(batch['label_ids']).cuda(
                    #     self.gpu, non_blocking=True)
                    inputs = batch[0].cuda(self.gpu, non_blocking=True)
                    input_mask = batch[1].cuda(self.gpu, non_blocking=True)
                    targets = batch[2].cuda(self.gpu, non_blocking=True)
            self.optimizer.zero_grad()
            outputs = self.model(
                inputs,
                attention_mask=input_mask if self.args.train.mask else None,
                labels=targets)
            loss = outputs[0]
            # loss = self.criterion(outputs, targets)
            # if len(self.gpu) > 1:
            #     loss = loss.mean()
            # if self.args.train.gradient_accumulation_steps > 1:
            # loss /= self.args.train.gradient_accumulation_steps
            train_loss += loss.item()
            loss.backward()
            if self.args.train.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.args.train.clip_grad)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        return train_loss / (step + 1)

    def validate(self, epoch: int) -> float:
        self.model.eval()
        val_loss = 0
        for step, batch in enumerate(
                tqdm.tqdm(self.val_loader, desc=f'Epoch {epoch} | Validate')):
            if self.gpu is not None and self.device == 'cuda':
                if len(batch) == 1:
                    # inputs = torch.LongTensor(batch['input_ids']).cuda(
                    #     self.gpu, non_blocking=True)
                    # input_mask = torch.LongTensor(batch['attention_mask']).cuda(
                    #     self.gpu, non_blocking=True)
                    # targets = inputs
                    inputs = batch.cuda(self.gpu, non_blocking=True)
                    targets = inputs
                if len(batch) == 2:
                    # inputs = torch.LongTensor(batch['input_ids']).cuda(
                    #     self.gpu, non_blocking=True)
                    # input_mask = torch.LongTensor(batch['attention_mask']).cuda(
                    #     self.gpu, non_blocking=True)
                    # targets = inputs
                    inputs = batch[0].cuda(self.gpu, non_blocking=True)
                    input_mask = batch[1].cuda(self.gpu, non_blocking=True)
                    targets = inputs
                else:
                    # inputs = torch.LongTensor(batch['input_ids']).cuda(
                    #     self.gpu, non_blocking=True)
                    # input_mask = torch.LongTensor(batch['attention_mask']).cuda(
                    #     self.gpu, non_blocking=True)
                    # targets = torch.LongTensor(batch['label_ids']).cuda(
                    #     self.gpu, non_blocking=True)
                    inputs = batch[0].cuda(self.gpu, non_blocking=True)
                    input_mask = batch[1].cuda(self.gpu, non_blocking=True)
                    targets = batch[2].cuda(self.gpu, non_blocking=True)
            with torch.no_grad():
                outputs = self.model(
                    inputs, attention_mask=input_mask, labels=targets)
            loss = outputs[0]
            # if len(self.gpu) > 1:
            #     loss = loss.mean()
            val_loss += loss.item()
        return val_loss / (step + 1)


def main(args: Namespace) -> None:
    training_agent = TrainingAgent(args=args)
    training_agent.train()
