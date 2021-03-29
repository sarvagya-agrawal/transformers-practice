"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from typing import List, Union, Dict, Tuple
from pathlib import PosixPath

import torch

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
# from transformers import LineByLineTextDataset
from torch.utils.data import Dataset, DataLoader, \
    RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=512):
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (
                len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(
            lines, add_special_tokens=True,
            truncation=True,
            max_length=block_size)

    def __len__(self):
        return len(self.examples['input_ids'])

    def __getitem__(self, i):
        return (torch.LongTensor(self.examples['input_ids'][i]),
                torch.LongTensor(self.examples['attention_mask'][i]))


def load_data(src_fname: PosixPath,
              tokenizer: PreTrainedTokenizerBase,
              max_length: int,
              batch_size: int,
              cache_dir: PosixPath,
              tgt_fname: PosixPath = None,
              overwrite_cache: bool = False,
              num_workers: int = 4,
              split='train') -> None:
    if src_fname.suffix not in set(['.txt']) or not src_fname.exists():
        raise ValueError(f"Unknown src file {src_fname}. Files must be",
                         "line-by-line .txt files")
    if tgt_fname is not None and (tgt_fname.suffix not in set(['.txt']) or
                                  not tgt_fname.exists()):
        raise ValueError(f"Unknown tgt file {tgt_fname}. Files must be",
                         "line-by-line .txt files")
    if tgt_fname is not None:
        pass
    else:
        # dataset = load_dataset('text',
        #                        data_files={split: str(src_fname)},
        #                        cache_dir=cache_dir, split=split)
        dataset = LineByLineTextDataset(tokenizer, file_path=str(src_fname),
                                        block_size=max_length)

    # def tokenize(examples: List[str]) -> Tuple[List[int], None]:
    #     examples["text"] = [line for line in examples["text"] if len(line) > 0
    #                         and not line.isspace()]
    #     tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    #     return tokenizer(
    #         examples['text'],
    #         add_special_tokens=True,
    #         padding='max_length',
    #         max_length=max_length,
    #         truncation=True,)

    # def collate(examples):
    #     examples[0] = pad_sequence(
    #         examples[0], batch_first=True,
    #         padding_value=tokenizer.pad_token_id if
    #         tokenizer._pad_token is not None else 0.)
    #     return examples
    # dataset = dataset.map(
    #     tokenize,
    #     batched=True,
    #     remove_columns=['text'],
    #     num_proc=num_workers,
    #     load_from_cache_file=not overwrite_cache
    # )
    sampler = RandomSampler(dataset) if split == 'train'\
        else SequentialSampler(dataset)
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        # collate_fn=collate,
        num_workers=num_workers)


def read_lines(filename: Union[str, PosixPath]) -> List[str]:
    """Read file and split into lines"""
    lines = open(filename).read().strip().split('\n')
    return [line for line in lines if (len(line) > 0 and not line.isspace())]


def vocabulary_indices(vocabulary: List[str]) -> Dict[str, int]:
    return {word: i for i, word in enumerate(sorted(list(vocabulary)))}
