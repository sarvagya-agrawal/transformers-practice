"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from collections import defaultdict
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Dict

import math
import time

from ray.tune.schedulers import ASHAScheduler
from accelerate import Accelerator
from transformers import set_seed
from ray.tune import CLIReporter
from adas import Adas
from ray import tune

import transformers
import datasets

import pandas as pd
import numpy as np
import torch
import tqdm

from ..optim import get_optimizer_scheduler
from ..utils import pretty_print_namespace
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
                 accelerator,
                 train_state: Dict[str, int] = None) -> None:
        self.args = args
        self.model = model
        self.tokenier = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.val_loader, self.val_dataset = val_data
        self.train_loader, self.train_dataset = train_data

        # self.stats = defaultdict(dict)
        self.output_dir = create_dir(Path(args.io.output))
        self.checkpoint_dir = create_dir(Path(args.io.checkpoint))
        self.cache_dir = create_dir(Path(args.io.cache_dir))
        self.start_epoch = \
            train_state['epoch'] if train_state is not None else 0
        self.start_trial = \
            train_state['trial'] if train_state is not None else 0
        self.stats = train_state['stats'] if train_state is not None else \
            defaultdict(dict)
        logger.info("Training Agent initialized")

    def save_stats(self) -> None:
        pd.DataFrame(data=self.stats).to_csv(self.output_dir / 'stats.csv')

    def save_checkpoint(self, trial: int, epoch: int, stamp: str) -> None:
        logger.info(f"Saving checkpoint {stamp}")
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped_model, 'save_pretrained'):
            unwrapped_model.save_pretrained(
                str(self.checkpoint_dir / stamp),
                save_function=self.accelerator.save)
        else:
            torch.save(unwrapped_model.state_dict(),
                       str(self.checkpoint_dir / stamp))
        # self.tokenizer.save_pretrained(str(self.checkpoint / stamp))
        # self.tokenizer.save_vocabulary(str(self.checkpoint / stamp))
        torch.save({
            'args': self.args,
            'epoch': epoch,
            'trial': trial,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is
            not None else None,
            'stats': self.stats},
            str(self.checkpoint_dir / stamp / 'train_state.pt'))

    def train(self) -> None:
        steps_completed = 0
        best_ppl = np.Inf
        last_step = len(self.train_loader) - 1
        for trial in range(self.start_trial, self.args.train.num_trials):
            progress_bar = tqdm.tqdm(
                range(self.args.train.max_train_steps),
                disable=not self.accelerator.is_local_main_process)
            if trial != self.start_trial:
                self.start_epoch = 0
            # elif self.start_epoch != 0:
            #     steps = int(self.args.train.max_train_steps /
            #                 self.args.train.max_epochs) * (self.start_epoch)
            #     for i in range(steps):
            #         progress_bar.update(1)

            for epoch in range(self.start_epoch, self.args.train.max_epochs):
                start_time = time.time()
                self.model.train()
                # train_loss, train_ppl = self.epoch_iteration(steps_completed)
                losses = list()
                for step, batch in enumerate(self.train_loader):
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
                        steps_completed += 1
                        progress_bar.update(1)
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
                if self.args.train.ray_tune:
                    tune.report(loss=val_ppl)
                if isinstance(self.optimizer.optimizer, Adas):
                    self.optimizer.optimizer.epoch_step(epoch)
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
        return {'loss': val_ppl}

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
        # logger.info(f"epoch {epoch}: perplexity: {perplexity}")
        return perplexity


def main(args: Namespace) -> None:
    if args.train.ray_tune:
        gpus_per_trial = torch.cuda.device_count()
        config = dict()
        for k, v in args.train.optimizer_kwargs.items():
            if not isinstance(v, list):
                continue
            if v[0] == 'loguniform':
                config[k] = tune.loguniform(float(v[1]), float(v[2]))
            elif v[0] == 'sample_from':
                config[k] = tune.sample_from(
                    lambda _: np.random.randint(int(v[1]), int(v[2])))
            elif v[0] == 'choice':
                config[k] = tune.choice([float(_) for _ in v[1:]])
            else:
                raise ValueError("Uknown RayTune parameter config")
        reporter = CLIReporter(
            # parameter_columns=["l1", "l2", "lr", "batch_size"],
            metric_columns=["loss", "training_iteration"])
        args.io.save_freq = args.train.max_epochs + 1
        hpo_scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=args.train.max_epochs,
            grace_period=1,
            reduction_factor=2)
        result = tune.run(
            partial(train_entry, args=args),
            resources_per_trial={"gpu": gpus_per_trial},
            config=config,
            num_samples=args.train.ray_tune_samples,
            scheduler=hpo_scheduler,
            progress_reporter=reporter,
            checkpoint_at_end=True,
            local_dir=str(Path(args.logs).parent))
        best_trial = result.get_best_trial("loss", "min", "last")
        logger.info("Best trial config: {}".format(best_trial.config))
        logger.info("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
    else:
        train_entry(None, args)


def train_entry(config, args: Namespace) -> None:
    accelerator = Accelerator()
    pretty_print_namespace('main', args)
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    train_state = None
    if args.io.resume is not None:
        path = Path(args.io.resume)
        if not path.exists():
            logger.warning("Unknown resume. Starting from scratch.")
        else:
            train_state = torch.load(str(path / 'train_state.pt'))
            resume = args.io.resume
            args = train_state['args']
            args.train.model = resume
            args.train.model_pretrained = True
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
    args.train.max_train_steps = max_train_steps = int(
        args.train.max_epochs * num_update_steps_per_epoch)

    if config is not None:
        for k, v in config.items():
            args.train.optimizer_kwargs[k] = v
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
    if train_state is not None:
        optimizer.load_state_dict(train_state['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(train_state['scheduler'])
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
        accelerator=accelerator,
        train_state=train_state)
    return agent.train()
