"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from typing import List, Union, Dict
from pathlib import PosixPath, Path

import numpy as np
import torch

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import default_data_collator, \
    DataCollatorForSeq2Seq
# from transformers import LineByLineTextDataset
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from ..utils.logging import logger


def right_shift(start_token_id: int,
                pad_token_id: int,
                input_ids: torch.Tensor) -> torch.Tensor:
    # shift inputs to the right
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = start_token_id

    assert pad_token_id is not None, \
        "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    assert torch.all(shifted_input_ids >= 0).item(
    ), "Verify that `shifted_input_ids` has only positive values"

    return shifted_input_ids


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=512):
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (
                len(line) > 0 and not line.isspace())]

        tokenized = tokenizer.batch_encode_plus(
            lines,
            add_special_tokens=True,
            padding=True,
            # return_tensors='pt',
            return_attention_mask=True,
            truncation=True,
            max_length=max_length)
        self.examples = tokenized['input_ids']
        self.masks = tokenized['attention_mask']

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # return (torch.LongTensor(self.examples['input_ids'][i]),
        #         torch.LongTensor(self.examples['attention_mask'][i]))
        # return torch.LongTensor(self.examples[i])
        return (torch.LongTensor(self.examples[i]),
                torch.LongTensor(self.masks[i]))


def extend_vocabulary(tokenizer, fname: PosixPath) -> None:
    if fname.suffix not in set(['.txt']) or not fname.exists():
        raise ValueError(f"Unknown src file {fname}. Files must be",
                         " .txt line by line files")
    vocab = [line.strip() for line in fname.open('r').readlines()]
    tokenizer.add_tokens(vocab)


def load_data(train_src: str,
              val_src: str,
              tokenizer: PreTrainedTokenizerBase,
              model: torch.nn.Module,
              max_src_length: int,
              max_tgt_length: int,
              block_size: int,
              batch_size: int,
              batch_size_eval: int,
              task: str,
              cache_dir: PosixPath,
              data_name: str = None,
              data_config: str = None,
              val_split: float = 0.0,
              max_train_samples: Union[int, float] = -1,
              max_val_samples: Union[int, float] = -1,
              ignore_pad_for_loss: bool = True,
              pad_to_max_length: bool = True,
              overwrite_cache: bool = False,
              num_workers: int = 4,
              prefix: str = '',
              ) -> None:
    if data_name is not None and data_config is not None:
        raw_datasets = load_dataset(
            data_name, data_config)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_name,
                data_config,
                split=f"train[:{val_split}%]",
            )
            raw_datasets["train"] = load_dataset(
                data_name,
                data_config,
                split=f"train[{val_split}%:]",
            )
    else:
        data_files = dict()
        data_files["train"] = train_src
        if val_src is not None:
            data_files["validation"] = val_src
        extension = Path(train_src).suffix[1:]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files)
    column_names = raw_datasets['train'].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    padding = "max_length" if pad_to_max_length else False
    if task == 'clm':
        if block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 1024:
                logger.warning(
                    "The tokenizer picked seems to have a very large" +
                    f" `model_max_length` ({tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value" +
                    " by passing --block_size xxx.")
            block_size = 1024
        else:
            if block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({block_size}) is larger " +
                    "than the maximum length for the model"
                    f"({tokenizer.model_max_length}). " +
                    f"Using block_size={tokenizer.model_max_length}.")
            block_size = min(block_size, tokenizer.model_max_length)

    def preprocess_data(examples):
        if task == 'nmt':
            inputs = [prefix + ex["source"]
                      for ex in examples["nmt"]]
            targets = [ex["target"] for ex in examples["nmt"]]
            model_inputs = tokenizer(inputs, max_length=max_src_length,
                                     padding=padding, truncation=True)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_tgt_length,
                                   padding=padding, truncation=True)

                if padding == 'max_length' and ignore_pad_for_loss:
                    labels['input_ids'] = [
                        [tok if tok != tokenizer.pad_token_id else -100 for
                            tok in label] for label in labels['input_ids']]
            model_inputs['labels'] = labels['input_ids']
            return model_inputs
        else:
            # Chunking handled in group texts: ignore warnings
            return tokenizer(examples[text_column_name])

    if max_train_samples > 0:
        if np.less(max_train_samples, 1):
            max_train_samples = int(
                max_train_samples * len(raw_datasets['train']))
        raw_datasets['train'] = raw_datasets['train'].select(
            range(min(max_train_samples,
                      len(raw_datasets['train']))))
    if max_val_samples > 0:
        if np.less(max_val_samples, 1):
            max_val_samples = int(
                max_val_samples * len(raw_datasets['validation']))
        raw_datasets['validation'] = raw_datasets['validation'].select(
            range(min(max_val_samples,
                      len(raw_datasets['validation']))))
    processed_datasets = raw_datasets.map(
        preprocess_data,
        batched=True,
        num_proc=num_workers,
        remove_columns=column_names,
        load_from_cache_file=not overwrite_cache)
    if task == 'clm':
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
            group_texts, batched=True, num_proc=num_workers,
            load_from_cache_file=not overwrite_cache)
    train_dataset = processed_datasets['train']
    val_dataset = processed_datasets['validation']

    if pad_to_max_length or task == 'clm':
        data_collator = default_data_collator
    else:
        label_pad_tok_id = -100 if ignore_pad_for_loss else \
            tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=label_pad_tok_id)
    train_loader = DataLoader(train_dataset, shuffle=True,
                              collate_fn=data_collator,
                              batch_size=batch_size,)
    val_loader = DataLoader(val_dataset, collate_fn=data_collator,
                            batch_size=batch_size_eval,)
    return train_loader, train_dataset, val_loader, val_dataset


def read_lines(filename: Union[str, PosixPath]) -> List[str]:
    """Read file and split into lines"""
    lines = open(filename).read().strip().split('\n')
    return [line for line in lines if (len(line) > 0 and not line.isspace())]


def vocabulary_indices(vocabulary: List[str]) -> Dict[str, int]:
    return {word: i for i, word in enumerate(sorted(list(vocabulary)))}
