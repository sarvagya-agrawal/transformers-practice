"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from typing import Dict, Any, List
from argparse import Namespace
from pathlib import Path

import logging
import random
import math
import time

from transformers import set_seed, AutoConfig, \
    AutoTokenizer, AutoModelForSeq2SeqLM, \
    DataCollatorForSeq2Seq, default_data_collator, \
    AutoModelForCausalLM, AdamW, get_scheduler
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator
from datasets import load_dataset

from adas import Adas
import transformers
import datasets

import pandas as pd
import numpy as np
import torch
import tqdm

# from ..data.utils import extend_vocabulary, load_data, right_shift
from ..models import get_tokenizer, get_model
from ..utils.logging import logger  # , mp_logger
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
        self.cache_dir = create_dir(Path(self.args.io.cache_dir))
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
        # if not self.args.train.tokenizer_pretrained:
        #     if self.args.rank == 0 or not self.args.distributed:
        #         (self.cache_dir /
        #          f'{self.args.train.tokenizer}_custom.json').unlink()
        self.tokenizer = get_tokenizer(
            self.args.train.tokenizer, self.args.io.cache_dir,
            dataset=self.args.data.name,
            pretrained=self.args.train.tokenizer_pretrained,
            datasets=self.args.data.tokenizer_files,
            task=self.args.data.task)
        # if self.args.rank == 0:
        #     dist.barrier()
        logger.info("Extending vocab...")
        # extend_vocabulary(self.tokenizer, fname=Path(self.args.data.vocab))
        # extend_vocabulary(self.tokenizer, fname=Path(valid_src))
        self.load_data()

    def load_data(self) -> None:
        self.train_loader, self.train_sampler = load_data(
            fname=Path(self.args.data.train_src),
            tokenizer=self.tokenizer,
            model=self.model,
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
            model=self.model,
            cache_dir=self.args.io.cache_dir,
            task=self.args.data.task,
            max_samples=self.args.data.max_val_samples,
            max_length=self.args.train.max_length,
            batch_size=self.args.train.batch_size,
            num_workers=self.args.data.num_workers,
            overwrite_cache=self.args.data.overwrite_cache,
            split='val',
            distributed=self.args.distributed)

    def reset(self) -> None:
        logger.info("Starting trial reset...")
        if self.args.distributed:
            self.args.train.batch_size = int(self.args.train.batch_size /
                                             self.args.ngpus_per_node)
            self.args.data.num_workers = int((self.args.data.num_workers +
                                              self.args.ngpus_per_node - 1) /
                                             self.args.ngpus_per_node)

        logger.info(f"Grabbing tokenizer: {self.args.train.tokenizer}")
        logger.info("Extending vocab...")
        logger.info("Loading model...")
        # if self.args.rank not in set([-1, 0]):
        #     dist.barrier()
        if self.args.io.resume is not None:
            self.load_checkpoint(Path(self.args.io.resume))
        else:
            self.model = get_model(self.args.train.model,
                                   cache_dir=self.args.io.cache_dir,
                                   tokenizer_len=len(self.tokenizer),
                                   pretrained=self.args.train.model_pretrained,
                                   task=self.args.data.task)
        logger.info("Loading train dataset...")
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
        # self.load_data()
        self.optimizer, self.scheduler = get_optimizer_scheduler(
            optim_method=self.args.train.optimizer,
            scheduler_method=self.args.train.scheduler,
            net_parameters=self.model.parameters(),
            max_epochs=self.args.train.max_epochs,
            train_loader_len=len(self.train_loader),
            optimizer_kwargs=self.args.train.optimizer_kwargs,
            scheduler_kwargs=self.args.train.scheduler_kwargs)
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
        self.args.train.model_pretrained = True
        train_state = torch.load(str(path / 'train_state.pt'),
                                 map_location=self.device)
        # train_state['args'].io.resume = self.args.io.resume
        # if self.args != train_state['args']:
        #     raise ValueError("Checkpoint args and config args don't match")
        # self.tokenizer = PreTrainedTokenizerFast(
        #     tokenizer_file=str(self.args.io.resume / 'tokenizer.json'))
        self.model = get_model(
            self.args.train.model,
            cache_dir=self.args.io.cache_dir,
            tokenizer_len=len(self.tokenizer),
            pretrained=True,
            weights=str(path),
            task=self.args.data.task)
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
        # self.load_data()
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
        # self.tokenizer.save_pretrained(str(self.checkpoint_dir / stamp))
        # self.tokenizer.save_vocabulary(str(self.checkpoint_dir / stamp))
        # self.tokenizer.save(str(self.checkpoint_dir / 'tokenizer.json'))
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
    accelerator = Accelerator()
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if args.data.data_name is not None and args.data.data_config is not None:
        raw_datasets = load_dataset(
            args.data.data_name, args.data.data_config)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.data_name,
                args.data_config,
                split=f"train[:{args.data.val_split}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.data_name,
                args.data_config,
                split=f"train[{args.data.val_split}%:]",
            )
    else:
        data_files = dict()
        data_files["train"] = args.data.train_src
        if args.data.val_src is not None:
            data_files["validation"] = args.data.val_src
        extension = Path(args.data.train_src).suffix[1:]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files)

    if args.train.hf_model_config is not None:
        config = AutoConfig.from_pretrained(args.train.hf_model_config) if \
            args.train.hf_model_config_pretrained else \
            AutoConfig(args.train.hf_model_config)
    else:
        config = AutoConfig.from_pretrained(args.train.model) if \
            args.train.hf_model_config_pretrained else \
            AutoConfig(args.train.model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.train.tokenizer, use_fast=True) if \
        args.train.tokenizer_pretrained else None

    if args.data.task == 'nmt':
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.train.model, config=config) if args.train.model_pretrained \
            else AutoModelForSeq2SeqLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.train.model, config=config) if args.train.model_pretrained \
            else AutoModelForCausalLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if args.data.task == 'nmt':
        if model.config.decoder_start_token_id is None:
            raise ValueError("Decoder start token id is None")

    column_names = raw_datasets['train'].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    padding = "max_length" if args.data.pad_to_max_length else False
    if args.data.task == 'clm':
        if args.data.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 1024:
                logger.warning(
                    "The tokenizer picked seems to have a very large" +
                    f" `model_max_length` ({tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value" +
                    " by passing --block_size xxx.")
            block_size = 1024
        else:
            if args.data.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({args.data.block_size}) is larger " +
                    "than the maximum length for the model"
                    f"({tokenizer.model_max_length}). " +
                    f"Using block_size={tokenizer.model_max_length}.")
            block_size = min(args.data.block_size, tokenizer.model_max_length)

    def preprocess_data(examples):
        if args.data.task == 'nmt':
            inputs = [args.data.prefix + ex["source"]
                      for ex in examples["nmt"]]
            targets = [ex["target"] for ex in examples["nmt"]]
            model_inputs = tokenizer(inputs, max_length=args.max_src_length,
                                     padding=padding, truncation=True)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=args.max_tgt_length,
                                   padding=padding, truncation=True)

                if padding == 'max_length' and args.data.ignore_pad__for_loss:
                    labels['input_ids'] = [
                        [tok if tok != tokenizer.pad_token_id else -100 for
                            tok in label] for label in labels['input_ids']]
            model_inputs['labels'] = labels['input_ids']
            return model_inputs
        else:
            # Chunking handled in group texts: ignore warnings
            return tokenizer(examples[text_column_name])

    if args.data.max_train_samples > 0:
        raw_datasets['train'] = raw_datasets['train'].select(
            range(min(args.data.max_train_samples,
                      len(raw_datasets['train']))))
    if args.data.max_val_samples > 0:
        raw_datasets['validation'] = raw_datasets['validation'].select(
            range(min(args.data.max_train_samples,
                      len(raw_datasets['validation']))))
    processed_datasets = raw_datasets.map(
        preprocess_data,
        batched=True,
        num_proc=args.data.num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.data.overwrite_cache)
    if args.data.task == 'clm':
        def group_texts(examples):
            concatenated_examples = {
                k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i: i + block_size]
                    for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        processed_datasets = processed_datasets.map(
            group_texts, batched=True, num_proc=args.data.num_workers,
            load_from_cache_file=not args.data.overwrite_cache)
    train_dataset = processed_datasets['train']
    val_dataset = processed_datasets['validation']

    if args.data.pad_to_max_length or args.data.task == 'clm':
        data_collator = default_data_collator
    else:
        label_pad_tok_id = -100 if args.ignore_pad__for_loss else \
            tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=label_pad_tok_id)
    train_loader = DataLoader(train_dataset, shuffle=True,
                              collate_fn=data_collator,
                              batch_size=args.train.batch_size,)
    val_loader = DataLoader(val_dataset, collate_fn=data_collator,
                            batch_size=args.train.batch_size_eval,)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)],
            "weight_decay": args.train.optimizer_kwargs['weight_decay'] if
            'weight_decay' in args.train.optimizer_kwargs.keys() else 0.0
        },
        {
            "params": [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.train.optimizer == 'Adas':
        optimizer_grouped_parameters.append({
            'all_params': list(model.parameters())})
        optimizer = Adas(optimizer_grouped_parameters,
                         params_dict=True,
                         **args.train.optimizer_kwargs)
    else:
        optimizer = AdamW(optimizer_grouped_parameters,
                          **args.train.optimizer_kwargs)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # steps = num iterations per epoch (len(dataset) / epoch) basically
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / args.train.gradient_accumulation_steps)
    max_train_steps = args.train.max_epochs * num_update_steps_per_epoch
    scheduler = None
    if args.train.optimizer != 'Adas':
        scheduler = get_scheduler(
            name=args.train.scheduler,
            optimizer=optimizer,
            num_training_steps=max_train_steps,
            **args.train.scheduler_kwargs)
    metric = None

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    progress_bar = tqdm.tqdm(range(max_train_steps),
                             disable=not accelerator.is_local_main_process)
    steps_completed = 0
    last_step = len(train_loader) - 1
    for epoch in range(args.train.max_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.train.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.train.gradient_accumulation_steps == 0 or \
                    step == last_step:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                steps_completed += 1
            else:
                continue
                if isinstance(optimizer, Adas):
                    optimizer.step()
            if steps_completed >= max_train_steps:
                break
        if isinstance(optimizer, Adas):
            optimizer.epoch_step()

        model.eval()
        losses = list()
        for step, batch in enumerate(val_loader):
            with torch.no_grad():
                if args.data.task == 'nmt':
                    generated_tokens = \
                        accelerator.unwrap_model(model).generate(
                            batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            max_length=args.data.max_tgt_length,
                            num_beams=args.num_beams)
                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1,
                        pad_index=tokenizer.pad_token_id)
                    labels = batch["labels"]
                    if not args.pad_to_max_length:
                        labels = accelerator.pad_across_processes(
                            batch["labels"], dim=1,
                            pad_index=tokenizer.pad_token_id)

                    generated_tokens = accelerator.gather(
                        generated_tokens).cpu().numpy()
                    labels = accelerator.gather(labels).cpu().numpy()

                    if args.ignore_pad_token_for_loss:
                        # Replace -100 in the labels as we can't decode them.
                        labels = np.where(labels != -100, labels,
                                          tokenizer.pad_token_id)

                    decoded_preds = tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True)
                    decoded_labels = tokenizer.batch_decode(
                        labels, skip_special_tokens=True)

                    decoded_preds, decoded_labels = postprocess_text(
                        decoded_preds, decoded_labels)

                    metric.add_batch(predictions=decoded_preds,
                                     references=decoded_labels)
                else:
                    outputs = model(**batch)
                    loss = outputs.loss
                    losses.append(accelerator.gather(
                        loss.repeat(args.train.batch_size_eval)))
        losses = torch.cat(losses)
        losses = losses[: len(val_dataset)]
        perplexity = math.exp(torch.mean(losses))
        if args.data.task == 'nmt':
            eval_metric = metric.compute()
            logger.info({"bleu": eval_metric["score"]})
        logger.info(f"epoch {epoch}: perplexity: {perplexity}")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.io.checkpoint, save_function=accelerator.save)
