"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from typing import List, Union, Dict, Tuple
from pathlib import PosixPath

from transformers import BertTokenizer, OpenAIGPTTokenizer
from torch.utils.data import TensorDataset, DataLoader, \
    RandomSampler, SequentialSampler

import torch

from .tokenizers import Tokenizer


def tokenize(lines: List[str],
             tokenizer_name: str,
             max_length: int,
             lower_case: bool = True) -> Tuple[List[int], None]:
    toks = ['bert-base-uncased']
    if tokenizer_name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(
            tokenizer_name, do_lower_case=lower_case)
        tokenized_inputs = tokenizer(
            lines,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt')
    elif tokenizer_name == 'openai-gpt':
        tokenizer = OpenAIGPTTokenizer.from_pretrained(
            tokenizer_name, do_lower_case=lower_case)
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        tokenized_inputs = tokenizer(
            lines,
            add_special_tokens=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt')
    else:
        raise ValueError(f'Unknown tokenizer. Must be one of {toks}')
    return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask']


def load_data(src_fname: PosixPath,
              tgt_fname: PosixPath,
              tokenizer_name: str,
              max_length: int,
              batch_size: int,
              num_workers: int = 4,
              lower_case: bool = True) -> None:
    src_data = read_lines(src_fname)
    tgt_data = read_lines(src_fname)
    src_ids, src_attention_mask = tokenize(src_data, tokenizer_name,
                                           max_length, lower_case)
    tgt_tensors = torch.tensor(tgt_data)
    data = TensorDataset(src_ids, src_attention_mask, tgt_tensors)
    data_sampler = RandomSampler(data)
    data_loader = DataLoader(data, sampler=data_sampler,
                             batch_size=batch_size, num_workers=num_workers)
    return data_loader


def read_lines(filename: Union[str, PosixPath]) -> List[str]:
    """Read file and split into lines"""
    return open(filename).read().strip().split('\n')


def vocabulary_indices(vocabulary: List[str]) -> Dict[str, int]:
    return {word: i for i, word in enumerate(sorted(list(vocabulary)))}
