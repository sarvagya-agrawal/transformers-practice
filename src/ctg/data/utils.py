"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from typing import List, Union, Dict, Tuple
from pathlib import PosixPath

import torch
import numpy as np

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
# from transformers import LineByLineTextDataset
from torch.utils.data import Dataset, DataLoader, \
    RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset


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


def load_data(src_fname: PosixPath,
              tokenizer: PreTrainedTokenizerBase,
              max_length: int,
              batch_size: int,
              cache_dir: PosixPath,
              max_samples: int = -1,
              overwrite_cache: bool = False,
              num_workers: int = 4,
              split: str = 'train',
              prefix: str = '',
              distributed: bool = False) -> None:
    if src_fname.suffix not in set(['.json']) or not src_fname.exists():
        raise ValueError(f"Unknown src file {src_fname}. Files must be",
                         " .json files")
    dataset = load_dataset('json',
                           data_files={split: str(src_fname)},
                           field='data',
                           cache_dir=cache_dir,
                           split=split)
    # dataset = LineByLineTextDataset(tokenizer, file_path=str(src_fname),
    #                                 max_length=max_length)

    def tokenize(examples: List[str]) -> Tuple[List[int], None]:
        inputs = examples["source"]
        inputs = [prefix + i for i in inputs]
        inputs = tokenizer(
            inputs,
            add_special_tokens=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='np',
            truncation=True)
        if "target" in examples.keys():
            targets = examples["target"]
            with tokenizer.as_target_tokenizer():
                targets = tokenizer(targets,
                                    add_special_tokens=True,
                                    padding='max_length',
                                    max_length=max_length,
                                    return_tensors='np',
                                    truncation=True)
                # targets["input_ids"] = [
                #     [(_label if _label != tokenizer.pad_token_id else -1e6)
                #         for _label in label] for label in targets["input_ids"]]
            del inputs['attention_mask']
            inputs["labels"] = targets["input_ids"]
        return inputs

    def collate(examples):
        return pad_sequence(
            np.array(examples), batch_first=True,
            padding_value=tokenizer.pad_token_id if
            tokenizer._pad_token is not None else 0.)
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_workers,
        # batch_size=batch_size,
        load_from_cache_file=not overwrite_cache
    )
    dataset.set_format(type='torch')
    if not distributed:
        sampler = RandomSampler(dataset) if split == 'train'\
            else SequentialSampler(dataset)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=split == 'train')
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        # collate_fn=collate,
        pin_memory=False,
        num_workers=num_workers), sampler


def read_lines(filename: Union[str, PosixPath]) -> List[str]:
    """Read file and split into lines"""
    lines = open(filename).read().strip().split('\n')
    return [line for line in lines if (len(line) > 0 and not line.isspace())]


def vocabulary_indices(vocabulary: List[str]) -> Dict[str, int]:
    return {word: i for i, word in enumerate(sorted(list(vocabulary)))}
