"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from collections import defaultdict
from argparse import Namespace
from pathlib import Path

import math
import time

from transformers import set_seed
from accelerate import Accelerator

import transformers
import datasets

import pandas as pd
import numpy as np
import torch
import tqdm

from ..optim import get_optimizer_scheduler
from ..models import get_model_tokenizer
from ..data.utils import load_data
from ..utils.logging import logger  # , mp_logger
from ..utils.io import create_dir


class TrainingAgent:
    def __init__(self,
                 args: Namespace,
                 model,
                 tokenizer,
                 optimizer,
                 scheduler,
                 val_data,
                 train_data,
                 accelerator,) -> None:
        self.args = args
        self.model = model
        self.tokenier = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.val_loader, self.val_dataset = val_data
        self.train_loader, self.train_dataset = train_data

        print(args.io)
        self.stats = defaultdict(dict)
        self.output_dir = create_dir(Path(args.io.output))
        self.checkpoint_dir = create_dir(Path(args.io.checkpoint))
        self.cache_dir = create_dir(Path(args.io.cache_dir))
        logger.info("Training Agent initialized")

    def save_stats(self) -> None:
        pd.DataFrame(data=self.stats).to_csv(self.output_dir / 'stats.csv')

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
        return
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
        progress_bar = tqdm.tqdm(
            range(self.args.train.max_train_steps),
            disable=not self.accelerator.is_local_main_process)
        steps_completed = 0
        best_ppl = np.Inf
        last_step = len(self.train_loader) - 1
        for trial in range(self.args.train.num_trials):
            for epoch in range(self.args.train.max_epochs):
                start_time = time.time()
                self.model.train()
                # train_loss, train_ppl = self.epoch_iteration(steps_completed)
                losses = list()
                for step, batch in enumerate(self.train_loader):
                    if step > 2:
                        break
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    loss = loss / self.args.train.gradient_accumulation_steps
                    self.accelerator.backward(loss)
                    losses.append(self.accelerator.gather(
                        loss.repeat(self.args.train.batch_size)))
                    if step % self.args.train.gradient_accumulation_steps == 0\
                            or step == last_step:
                        self.optimizer.step()
                        if self.scheduler is not None:
                            self.scheduler.step()
                        self.optimizer.zero_grad()
                        progress_bar.update(1)
                        steps_completed += 1
                    else:
                        continue
                        if isinstance(self.optimizer, Adas):
                            self.optimizer.step()
                    if self.args.train.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.args.train.clip_grad)
                    if steps_completed >= self.args.train.max_train_steps:
                        break
                epoch_time = time.time()
                val_ppl = self.validate(trial, epoch)
                end_time = time.time()

                losses = torch.cat(losses)
                losses = losses[: len(self.val_dataset)]
                train_ppl = math.exp(torch.mean(losses))
                rem_time = (end_time - start_time) * \
                    self.args.train.max_epochs - 1 - epoch
                logger.info(f"T {trial} | E {epoch} | " +
                            f"E time: {epoch_time - start_time:.2f}s | " +
                            f"V time: {end_time - epoch_time:.2f}s | " +
                            f"Remaining time: {rem_time:.2f}s | " +
                            f"Train PPL {train_ppl} | " +
                            f"Val PPL {val_ppl}")
                self.stats[epoch]['train_ppl'] = train_ppl
                self.stats[epoch]['val_ppl'] = val_ppl
            if isinstance(self.optimizer, Adas):
                self.optimizer.epoch_step(epoch)
            if (epoch % self.args.io.save_freq == 0 and epoch > 0):
                self.save_checkpoint(trial=trial,
                                     epoch=epoch,
                                     stamp="latest")
            if np.less(val_ppl, best_ppl):
                best_ppl = val_ppl
                self.save_checkpoint(trial=trial,
                                     epoch=epoch,
                                     stamp="best")
            self.save_stats()

            # accelerator.wait_for_everyone()
            # unwrapped_model = accelerator.unwrap_model(model)
            # unwrapped_model.save_pretrained(
            #     args.io.checkpoint, save_function=accelerator.save)
            self.save_checkpoint(trial=trial,
                                 epoch=self.args.train.max_epochs - 1,
                                 stamp=f'final-{trial}')

    def evaluate(self):
        ...

    def validate(self, trial: int, epoch: int) -> float:
        self.model.eval()
        losses = list()
        for step, batch in enumerate(self.val_loader):
            with torch.no_grad():
                if self.args.data.task == 'nmt':
                    generated_tokens = \
                        self.accelerator.unwrap_model(self.model).generate(
                            batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            max_length=self.args.data.max_tgt_length,
                            num_beams=self.args.num_beams)
                    generated_tokens = self.accelerator.pad_across_processes(
                        generated_tokens, dim=1,
                        pad_index=self.tokenizer.pad_token_id)
                    labels = batch["labels"]
                    if not self.args.pad_to_max_length:
                        labels = self.accelerator.pad_across_processes(
                            batch["labels"], dim=1,
                            pad_index=self.tokenizer.pad_token_id)

                    generated_tokens = self.accelerator.gather(
                        generated_tokens).cpu().numpy()
                    labels = self.accelerator.gather(labels).cpu().numpy()

                    if self.args.ignore_pad_token_for_loss:
                        # Replace -100 in the labels as we can't decode them.
                        labels = np.where(labels != -100, labels,
                                          self.tokenizer.pad_token_id)

                    decoded_preds = self.tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True)
                    decoded_labels = self.tokenizer.batch_decode(
                        labels, skip_special_tokens=True)

                    def postprocess_text(preds, labels):
                        preds = [pred.strip() for pred in preds]
                        labels = [[label.strip()] for label in labels]

                        return preds, labels
                    decoded_preds, decoded_labels = postprocess_text(
                        decoded_preds, decoded_labels)

                    metric.add_batch(predictions=decoded_preds,
                                     references=decoded_labels)
                outputs = self.model(**batch)
                loss = outputs.loss
                losses.append(self.accelerator.gather(
                    loss.repeat(self.args.train.batch_size_eval)))
        losses = torch.cat(losses)
        losses = losses[: len(self.val_dataset)]
        perplexity = math.exp(torch.mean(losses))
        if self.args.data.task == 'nmt':
            eval_metric = metric.compute()
            logger.info({"bleu": eval_metric["score"]})
        logger.info(f"epoch {epoch}: perplexity: {perplexity}")
        return perplexity


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

    model, tokenizer = get_model_tokenizer(
        model_name=args.train.model,
        model_pretrained=args.train.model_pretrained,
        tokenizer_name=args.train.tokenizer,
        tokenizer_pretrained=args.train.tokenizer_pretrained,
        hf_model_config=args.train.hf_model_config,
        hf_model_config_pretrained=args.train.hf_model_config_pretrained,
        task=args.data.task)
    train_loader, train_dataset, val_loader, val_dataset = load_data(
        tokenizer=tokenizer,
        model=model,
        train_src=args.data.train_src,
        val_src=args.data.val_src,
        data_name=args.data.data_name,
        data_config=args.data.data_config,
        val_split=args.data.val_split,
        pad_to_max_length=args.data.pad_to_max_length,
        task=args.data.task,
        block_size=args.data.block_size,
        prefix=args.data.prefix,
        max_src_length=args.train.max_src_length,
        max_tgt_length=args.train.max_tgt_length,
        ignore_pad_for_loss=args.data.ignore_pad_for_loss,
        max_train_samples=args.data.max_train_samples,
        max_val_samples=args.data.max_val_samples,
        num_workers=args.data.num_workers,
        overwrite_cache=args.data.overwrite_cache,
        cache_dir=args.io.cache_dir,
        batch_size=args.train.batch_size,
        batch_size_eval=args.train.batch_size_eval,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / args.train.gradient_accumulation_steps)
    args.train.max_train_steps = max_train_steps = \
        args.train.max_epochs * num_update_steps_per_epoch
    optimizer, scheduler = get_optimizer_scheduler(
        optim_method=args.train.optimizer,
        scheduler_method=args.train.scheduler,
        params=list(model.parameters()),
        named_parameters=list(model.named_parameters()),
        max_epochs=args.train.max_epochs,
        max_train_steps=max_train_steps,
        optimizer_kwargs=args.train.optimizer_kwargs,
        scheduler_kwargs=args.train.scheduler_kwargs,
    )
    # Prepare everything with our `accelerator`.
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # steps = num iterations per epoch (len(dataset) / epoch) basically
    agent = TrainingAgent(
        args=args,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        val_data=(val_loader, val_dataset),
        train_data=(train_loader, train_dataset),
        accelerator=accelerator)
    agent.train()
