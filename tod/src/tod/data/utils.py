"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from typing import List, Union, Dict, Tuple
from pathlib import PosixPath

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import LineByLineTextDataset
from torch.utils.data import TensorDataset, DataLoader, \
    RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence

import torch


def tokenize(lines: List[str],
             tokenizer: PreTrainedTokenizerBase,
             max_length: int,) -> Tuple[List[int], None]:
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokenized_inputs = tokenizer(
        lines,
        add_special_tokens=True,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_tensors='pt')
    return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask']


def load_data(src_fname: PosixPath,
              tgt_fname: PosixPath,
              tokenizer: PreTrainedTokenizerBase,
              max_length: int,
              batch_size: int,
              num_workers: int = 4,
              split='train') -> None:
    src_data = read_lines(src_fname)
    tgt_data = read_lines(src_fname)
    src_ids, src_attention_mask = tokenize(src_data, tokenizer,
                                           max_length)
    if src_fname == tgt_fname:
        tgt_tensors = src_ids
    else:
        tgt_tensors, _ = tokenize(tgt_data, tokenizer, max_length)
    data = TensorDataset(src_ids, src_attention_mask, tgt_tensors)
    data_sampler = RandomSampler(
        data) if split == 'train' else SequentialSampler(data)

    # def collate(lines):
    #     if tokenizer._pad_token is None:
    #         return pad_sequence(lines, batch_first=True)
    #     return pad_sequence(
    #         lines, batch_first=True, padding_value=tokenizer.pad_token_id)
    data_loader = DataLoader(
        data, sampler=data_sampler, batch_size=batch_size,
        # num_workers=num_workers, collate_fn=collate)
        num_workers=num_workers)
    return data_loader


def read_lines(filename: Union[str, PosixPath]) -> List[str]:
    """Read file and split into lines"""
    lines = open(filename).read().strip().split('\n')
    return [line for line in lines if (len(line) > 0 and not line.isspace())]


def vocabulary_indices(vocabulary: List[str]) -> Dict[str, int]:
    return {word: i for i, word in enumerate(sorted(list(vocabulary)))}
